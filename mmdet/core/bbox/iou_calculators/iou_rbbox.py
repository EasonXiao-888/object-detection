from mmdet.ops.fcosr_tool import tools_cuda
import torch


def get_iou(rboxes_1: torch.Tensor, rboxes_2: torch.Tensor):
    rboxes_1 = rboxes_1.float()
    rboxes_2 = rboxes_2.float()
    if not rboxes_1.is_contiguous():
        rboxes_1 = rboxes_1.contiguous()
    if not rboxes_2.is_contiguous():
        rboxes_2 = rboxes_2.contiguous()
    # return fcosr_tools.compute_poly_iou(fcosr_tools.rbox2corner(rboxes_1, angle_positive), fcosr_tools.rbox2corner(rboxes_2, angle_positive))
    return tools_cuda.compute_rbox_iou(rboxes_1, rboxes_2)