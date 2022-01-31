
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from ..transforms_rbbox import *
import AerialDetection.DOTA_devkit.polyiou as polyiou
from .builder import IOU_CALCULATORS
from mmdet.core.bbox.ops import bbox_overlaps_cython

@IOU_CALCULATORS.register_module()
class rbbox_overlaps_cy:
    def rbbox_overlaps_cy_warp(rbboxes, query_boxes):
        # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps
        # import pdb
        # pdb.set_trace()
        box_device = query_boxes.device
        query_boxes_np = query_boxes.cpu().numpy().astype(np.float)
        rbboxes = rbboxes.cpu().numpy()

        # polys_np = RotBox2Polys(boxes_np)
        # TODO: change it to only use pos gt_masks
        # polys_np = mask2poly(gt_masks)
        # polys_np = np.array(Tuplelist2Polylist(polys_np)).astype(np.float)

        polys_np = RotBox2Polys(rbboxes).astype(np.float)
        query_polys_np = RotBox2Polys(query_boxes_np)

        h_bboxes_np = poly2bbox(polys_np)
        h_query_bboxes_np = poly2bbox(query_polys_np)

        # hious
        ious = bbox_overlaps_cython(h_bboxes_np, h_query_bboxes_np) 
        import pdb
        # pdb.set_trace()
        inds = np.where(ious > 0)
        for index in range(len(inds[0])):
            box_index = inds[0][index]
            query_box_index = inds[1][index]

            box = polys_np[box_index]
            query_box = query_polys_np[query_box_index]

        # calculate obb iou
        # import pdb
        # pdb.set_trace()
            overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
            ious[box_index][query_box_index] = overlap

        return torch.from_numpy(ious).to(box_device)

def rbbox_overlaps_cy_warp(rbboxes, query_boxes):
        # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps
        # import pdb
        # pdb.set_trace()

        # box_device = query_boxes.device
        # query_boxes_np = query_boxes.cpu().numpy().astype(np.float)

        # rbboxes = rbboxes.cpu()
        # rbboxes = rbboxes.numpy()
        # print(type(rbboxes),type(query_boxes))
        # polys_np = RotBox2Polys(boxes_np)
        # TODO: change it to only use pos gt_masks
        # polys_np = mask2poly(gt_masks)
        # polys_np = np.array(Tuplelist2Polylist(polys_np)).astype(np.float)

        polys_np = RotBox2Polys_torch(rbboxes)
        query_polys_np = RotBox2Polys_torch(query_boxes)

        h_bboxes_np = poly2bbox_torch(polys_np)
        h_query_bboxes_np = poly2bbox_torch(query_polys_np)

        # h_query_bboxes_np = torch.from_numpy(h_query_bboxes_np)
        # h_bboxes_np = torch.from_numpy(h_bboxes_np)

        # print(type(h_bboxes_np),type(h_query_bboxes_np))

        # hious
        ious = bbox_overlaps(h_bboxes_np, h_query_bboxes_np,mode='iou', is_aligned=False, eps=1e-6)
        # print(ious)
        # ious = bbox_overlaps_cython(h_bboxes_np, h_query_bboxes_np)
        # import pdb
        # pdb.set_trace()
        # # inds = np.where(ious > 0)
        # for index in range(len(inds[0])):
        #     box_index = inds[0][index]
        #     query_box_index = inds[1][index]

        #     box = polys_np[box_index]
        #     query_box = query_polys_np[query_box_index]

        # # calculate obb iou
        # # import pdb
        # # pdb.set_trace()
        #     overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
        #     ious[box_index][query_box_index] = overlap
        # ious=ious.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return ious

