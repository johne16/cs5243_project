# COCO

## Object detection annotation format

```
annotation{
"id": int,
"image_id": int,
"category_id": int,
"segmentation": RLE or [polygon],
"area": float,
"bbox": [x,y,width,height],
"iscrowd": 0 or 1,
}

categories[{
"id": int,
"name": str,
"supercategory": str,
}]
```

## Loading a class subset with FiftyOne

```
dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections", "segmentations"],
    classes=["person", "car"],
    only_matching=True,
    max_samples=50,
)
```

## Class subset

| id | name |
|---|---|
| 1 | person |
| 2 | bicycle |
| 3 | car |
| 4 | motorcycle |
| 5 | airplane |
| 6 | bus |
| 7 | train |
| 8 | truck |
| 9 | boat |
