// SPDX-License-Identifier: MIT OR Apache-2.0
//! Render-pass and dynamic-rendering recording on [`CommandBufferRecording`].

use super::*;

impl<'a> CommandBufferRecording<'a> {
    /// Record `vkCmdBeginRenderPass`: start a render pass instance
    /// targeting `framebuffer`. The clear values must contain one
    /// entry per render-pass attachment that uses
    /// `LoadOp::CLEAR` (or one entry per attachment, with the others
    /// ignored — the safest option).
    ///
    /// `clear_colors` are interpreted as `f32; 4` per slot.
    pub fn begin_render_pass(
        &mut self,
        render_pass: &RenderPass,
        framebuffer: &Framebuffer,
        clear_colors: &[[f32; 4]],
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBeginRenderPass
            .expect("vkCmdBeginRenderPass is required by Vulkan 1.0");

        let raw_clears: Vec<VkClearValue> = clear_colors
            .iter()
            .map(|c| VkClearValue {
                color: VkClearColorValue { float32: *c },
            })
            .collect();

        let info = VkRenderPassBeginInfo {
            sType: VkStructureType::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass: render_pass.handle,
            framebuffer: framebuffer.handle,
            renderArea: VkRect2D {
                offset: VkOffset2D { x: 0, y: 0 },
                extent: VkExtent2D {
                    width: framebuffer.width,
                    height: framebuffer.height,
                },
            },
            clearValueCount: raw_clears.len() as u32,
            pClearValues: if raw_clears.is_empty() {
                std::ptr::null()
            } else {
                raw_clears.as_ptr()
            },
            ..Default::default()
        };
        // VK_SUBPASS_CONTENTS_INLINE = 0
        // Safety: command buffer is recording, info+raw_clears live until call end.
        unsafe {
            cmd(
                self.buffer.handle,
                &info,
                VkSubpassContents::SUBPASS_CONTENTS_INLINE,
            )
        };
    }

    /// Like [`begin_render_pass`](Self::begin_render_pass), but accepts
    /// [`ClearValue`] entries so you can mix color and depth/stencil
    /// clears in a single render pass (e.g. one color attachment + one
    /// depth attachment).
    pub fn begin_render_pass_ext(
        &mut self,
        render_pass: &RenderPass,
        framebuffer: &Framebuffer,
        clear_values: &[ClearValue],
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBeginRenderPass
            .expect("vkCmdBeginRenderPass is required by Vulkan 1.0");

        let raw_clears: Vec<VkClearValue> = clear_values
            .iter()
            .map(|cv| match cv {
                ClearValue::Color(c) => VkClearValue {
                    color: VkClearColorValue { float32: *c },
                },
                ClearValue::DepthStencil { depth, stencil } => VkClearValue {
                    depthStencil: VkClearDepthStencilValue {
                        depth: *depth,
                        stencil: *stencil,
                    },
                },
            })
            .collect();

        let info = VkRenderPassBeginInfo {
            sType: VkStructureType::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass: render_pass.handle,
            framebuffer: framebuffer.handle,
            renderArea: VkRect2D {
                offset: VkOffset2D { x: 0, y: 0 },
                extent: VkExtent2D {
                    width: framebuffer.width,
                    height: framebuffer.height,
                },
            },
            clearValueCount: raw_clears.len() as u32,
            pClearValues: if raw_clears.is_empty() {
                std::ptr::null()
            } else {
                raw_clears.as_ptr()
            },
            ..Default::default()
        };
        unsafe {
            cmd(
                self.buffer.handle,
                &info,
                VkSubpassContents::SUBPASS_CONTENTS_INLINE,
            )
        };
    }

    /// Record `vkCmdEndRenderPass`.
    pub fn end_render_pass(&mut self) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdEndRenderPass
            .expect("vkCmdEndRenderPass is required by Vulkan 1.0");
        // Safety: command buffer is in recording state.
        unsafe { cmd(self.buffer.handle) };
    }

    /// Record `vkCmdBeginRendering` (Vulkan 1.3 core, or
    /// `VK_KHR_dynamic_rendering` on 1.1/1.2).
    ///
    /// Dynamic rendering replaces `VkRenderPass` + `VkFramebuffer` with
    /// a single call that names the attachments and their load/store
    /// ops directly. The matching [`end_rendering`](Self::end_rendering)
    /// closes the rendering scope.
    ///
    /// Returns `MissingFunction("vkCmdBeginRendering")` if the feature /
    /// extension is not enabled on the device.
    pub fn begin_rendering(&mut self, info: RenderingInfo<'_>) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBeginRendering
            .ok_or(Error::MissingFunction("vkCmdBeginRendering"))?;

        let to_raw = |a: &RenderingAttachment<'_>| -> VkRenderingAttachmentInfo {
            let clear = match a.clear_value {
                Some(ClearValue::Color([r, g, b, al])) => VkClearValue {
                    color: VkClearColorValue {
                        float32: [r, g, b, al],
                    },
                },
                Some(ClearValue::DepthStencil { depth, stencil }) => VkClearValue {
                    depthStencil: VkClearDepthStencilValue { depth, stencil },
                },
                None => unsafe { std::mem::zeroed() },
            };
            VkRenderingAttachmentInfo {
                sType: VkStructureType::STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                pNext: std::ptr::null(),
                imageView: a.view.handle,
                imageLayout: a.layout.0,
                resolveMode: unsafe { std::mem::zeroed() },
                resolveImageView: 0,
                resolveImageLayout: VkImageLayout::IMAGE_LAYOUT_UNDEFINED,
                loadOp: a.load_op.0,
                storeOp: a.store_op.0,
                clearValue: clear,
            }
        };

        let color_raw: Vec<VkRenderingAttachmentInfo> =
            info.color_attachments.iter().map(to_raw).collect();
        let depth_raw = info.depth_attachment.as_ref().map(to_raw);
        let stencil_raw = info.stencil_attachment.as_ref().map(to_raw);

        let raw = VkRenderingInfo {
            sType: VkStructureType::STRUCTURE_TYPE_RENDERING_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            renderArea: info.render_area,
            layerCount: info.layer_count,
            viewMask: info.view_mask,
            colorAttachmentCount: color_raw.len() as u32,
            pColorAttachments: if color_raw.is_empty() {
                std::ptr::null()
            } else {
                color_raw.as_ptr()
            },
            pDepthAttachment: depth_raw
                .as_ref()
                .map_or(std::ptr::null(), |a| a as *const _),
            pStencilAttachment: stencil_raw
                .as_ref()
                .map_or(std::ptr::null(), |a| a as *const _),
        };

        // Safety: command buffer recording; raw, color_raw, depth_raw,
        // stencil_raw all outlive the synchronous call.
        unsafe { cmd(self.buffer.handle, &raw) };
        Ok(())
    }

    /// Record `vkCmdEndRendering` — closes a scope opened by
    /// [`begin_rendering`](Self::begin_rendering).
    ///
    /// Returns `MissingFunction("vkCmdEndRendering")` if the
    /// corresponding feature / extension is not enabled.
    pub fn end_rendering(&mut self) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdEndRendering
            .ok_or(Error::MissingFunction("vkCmdEndRendering"))?;
        // Safety: command buffer is in recording state.
        unsafe { cmd(self.buffer.handle) };
        Ok(())
    }
}
