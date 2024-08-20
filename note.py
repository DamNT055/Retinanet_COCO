from pycocotools.cocoeval import COCOeval
import json

# Chuyển đổi kết quả dự đoán của bạn thành định dạng COCO
coco_results = []
for i, (images, targets) in enumerate(data_loader):
    images = list(image.to(device) for image in images)
    outputs = model(images)

    for j, output in enumerate(outputs):
        image_id = targets[j]["image_id"].item()
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            coco_results.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": [x_min, y_min, width, height],
                "score": score
            })

# Lưu kết quả dưới dạng file JSON
with open("coco_results.json", "w") as f:
    json.dump(coco_results, f)

# Sử dụng COCOeval để tính toán mAP
coco_gt = dataset.coco
coco_dt = coco_gt.loadRes("coco_results.json")
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
