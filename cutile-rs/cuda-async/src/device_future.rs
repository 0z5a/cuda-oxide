/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::device_operation::{DeviceOperation, ExecutionContext};
use crate::error::DeviceError;
use futures::task::AtomicWaker;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum DeviceFutureState {
    // The future was created with an error and will resolve immediately on first poll.
    Failed,
    // The stream operation has not yet been scheduled. No callback has been added.
    Idle,
    // The stream operation has been scheduled and a callback has been added to the stream.
    // The callback should be added such that it immediately succeeds the scheduled operation.
    Executing,
    // The callback has been fired, indicating the completion of the stream operation.
    Complete,
}

#[derive(Debug)]
pub struct StreamCallbackState {
    pub(crate) waker: AtomicWaker,
    pub(crate) complete: AtomicBool,
}

impl StreamCallbackState {
    pub fn new() -> Self {
        Self {
            waker: AtomicWaker::new(),
            complete: AtomicBool::new(false),
        }
    }
    pub fn signal(&self) {
        self.complete.store(true, Ordering::Relaxed);
        self.waker.wake();
    }
}

#[derive(Debug)]
pub struct DeviceFuture<T: Send, DO: DeviceOperation<Output = T>> {
    pub(crate) device_operation: Option<DO>,
    pub(crate) execution_context: Option<ExecutionContext>,
    pub(crate) result: Option<T>,
    pub(crate) error: Option<DeviceError>,
    pub(crate) state: DeviceFutureState,
    pub(crate) callback_state: Option<Arc<StreamCallbackState>>,
}

impl<T: Send, DO: DeviceOperation<Output = T>> DeviceFuture<T, DO> {
    pub fn new() -> Self {
        Self {
            execution_context: None,
            device_operation: None,
            state: DeviceFutureState::Idle,
            callback_state: None,
            result: None,
            error: None,
        }
    }

    /// Create a future that is pre-loaded with an error.
    ///
    /// On first poll it immediately returns `Poll::Ready(Err(error))`.
    /// This is used by `IntoFuture` implementations to surface scheduling
    /// failures without panicking.
    pub fn failed(error: DeviceError) -> Self {
        Self {
            execution_context: None,
            device_operation: None,
            state: DeviceFutureState::Failed,
            callback_state: None,
            result: None,
            error: Some(error),
        }
    }

    unsafe fn register_callback(
        &self,
        waker_state: Arc<StreamCallbackState>,
    ) -> Result<(), DeviceError> {
        let ctx = self
            .execution_context
            .as_ref()
            .ok_or(DeviceError::Internal(
                "Cannot execute future without setting stream on which to execute.".to_string(),
            ))?;
        ctx.get_cuda_stream().launch_host_function(move || {
            waker_state.signal();
        })?;
        Ok(())
    }
    fn execute(&mut self) -> Result<(), DeviceError> {
        let ctx = self
            .execution_context
            .as_ref()
            .ok_or(DeviceError::Internal(
                "Cannot execute future without setting stream on which to execute.".to_string(),
            ))?;
        // TODO (hme): We may need to hold a reference to device_operation,
        //  to ensure kernel launch structs (and their args) are dropped
        //  when the future completes vs. when this function completes.
        let operation = self.device_operation.take().ok_or(DeviceError::Internal(
            "Unable to execute future: No operation has been set.".to_string(),
        ))?;
        let out = unsafe { operation.execute(ctx) }?;
        self.result = Some(out);
        Ok(())
    }
}

impl<T: Send, DO: DeviceOperation<Output = T>> Unpin for DeviceFuture<T, DO> {}

impl<T: Send, DO: DeviceOperation<Output = T>> Future for DeviceFuture<T, DO> {
    type Output = Result<T, DeviceError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.state {
            DeviceFutureState::Failed => {
                self.state = DeviceFutureState::Complete;
                let error = self
                    .error
                    .take()
                    .expect("Failed state must carry an error.");
                return Poll::Ready(Err(error));
            }
            _ => {}
        }

        // If this is being polled, it needs a waker.
        if self.callback_state.is_none() {
            self.callback_state = Some(Arc::new(StreamCallbackState::new()));
        }
        let waker_state = self.callback_state.as_ref().cloned().expect("Impossible.");
        match self.state {
            DeviceFutureState::Idle => {
                // Initialize the waker.
                waker_state.waker.register(cx.waker());
                // Execute this future's operation.
                if let Err(e) = self.execute() {
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Err(e));
                }
                // Add the callback. We only want to do this once.
                if let Err(e) = unsafe { self.register_callback(waker_state.clone()) } {
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Err(e));
                }
                // Transition the future's state to "Executing."
                self.state = DeviceFutureState::Executing;
                Poll::Pending
            }
            DeviceFutureState::Executing => {
                // The future may have been polled by the waker firing or by some other mechanism.
                // Check if the complete flag has been set by the callback.
                if waker_state.complete.load(Ordering::Relaxed) {
                    self.state = DeviceFutureState::Complete;
                    // If the future was polled by some mechanism other than the waker,
                    // then the old waker still may fire, but the future will not be polled
                    // again if we return Poll::Ready.
                    return Poll::Ready(Ok(self
                        .result
                        .take()
                        .expect("Expected future result to be Some.")));
                }
                // The future is still incomplete. Update the waker to the latest context.
                waker_state.waker.register(cx.waker());
                // Check if the callback has fired after updating the waker.
                // If the callback triggers the old waker before the new waker is registered,
                // the newly registered waker will never be called.
                if waker_state.complete.load(Ordering::Relaxed) {
                    self.state = DeviceFutureState::Complete;
                    Poll::Ready(Ok(self
                        .result
                        .take()
                        .expect("Expected future result to be Some.")))
                } else {
                    Poll::Pending
                }
            }
            DeviceFutureState::Complete => {
                // We set the future's state to complete before returning Poll::Ready.
                // The executor *should* never poll this task again.
                panic!("Poll called after completion.");
            }
            DeviceFutureState::Failed => {
                // Already handled above; this arm is unreachable.
                unreachable!();
            }
        }
    }
}
