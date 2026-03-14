/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::device_context::with_deallocator_stream;
use crate::launch::{AsyncKernelLaunch, KernelArgument};
use cuda_core::free_async;
use cuda_core::sys::CUdeviceptr;
use std::marker::PhantomData;

#[derive(Debug, Copy, Clone)]
pub struct DevicePointer<T> {
    dtype: PhantomData<T>,
    pub dptr: CUdeviceptr,
}

unsafe impl<T> Send for DevicePointer<T> {}

impl<T> DevicePointer<T> {
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.dptr
    }
}

impl<T: Send + Sized> KernelArgument for DevicePointer<T> {
    fn push_arg(self, launcher: &mut AsyncKernelLaunch) {
        launcher.push_arg(Box::new(self.cu_deviceptr()));
    }
}

#[derive(Debug)]
pub struct DeviceBox<T: Send + ?Sized> {
    _device_id: usize,
    dtype: PhantomData<T>,
    cudptr: CUdeviceptr,
    len: usize,
}

unsafe impl<T: Send + ?Sized> Send for DeviceBox<T> {}
unsafe impl<T: Send + ?Sized> Sync for DeviceBox<T> {}

impl<T: Send + ?Sized> Drop for DeviceBox<T> {
    fn drop(&mut self) {
        unsafe {
            // Safety: The CUDA driver is guaranteed to complete any queued async operations.
            with_deallocator_stream(self._device_id, |stream| {
                free_async(self.cudptr, stream);
            })
            .expect(
                format!(
                    "Failed to free device pointer on device_id={}",
                    self._device_id
                )
                .as_str(),
            )
        }
    }
}

impl<DType: Send + Sized> DeviceBox<[DType]> {
    pub unsafe fn from_raw_parts(dptr: CUdeviceptr, len: usize, device_id: usize) -> Self {
        Self {
            dtype: PhantomData,
            cudptr: dptr,
            len,
            _device_id: device_id,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.cudptr
    }
    pub fn device_id(&self) -> usize {
        self._device_id
    }
    pub fn device_pointer(&self) -> DevicePointer<DType> {
        DevicePointer::<DType> {
            dtype: PhantomData,
            dptr: self.cudptr,
        }
    }
}
