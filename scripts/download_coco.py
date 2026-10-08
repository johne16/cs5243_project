import json
import random
from pathlib import Path

import fiftyone.utils.coco as fouc

classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
sampled_classes = ["bicycle", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
n_per_class = 3000
seed = 0

coco_dir = Path(__file__).resolve().parents[1] / "datasets" / "coco"
raw_dir = coco_dir / "raw"
scratch_dir = coco_dir / "scratch"
ann_dir = coco_dir / "annotations"


def load_raw_anns(split_name):
    with open(raw_dir / f"instances_{split_name}.json") as fp:
        return json.load(fp)


def get_img_ids_per_cat(raw_anns, cat_ids):
    img_ids_per_cat = {cat_id: set() for cat_id in cat_ids}
    for ann in raw_anns["annotations"]:
        if ann["category_id"] in cat_ids:
            img_ids_per_cat[ann["category_id"]].add(ann["image_id"])
    return img_ids_per_cat


def write_subset_anns(raw_anns, img_ids, cat_ids, split_name):
    subset_anns = dict(raw_anns)
    subset_anns["categories"] = [c for c in raw_anns["categories"] if c["id"] in cat_ids]
    subset_anns["images"] = [i for i in raw_anns["images"] if i["id"] in img_ids]
    subset_anns["annotations"] = [
        a for a in raw_anns["annotations"] if a["image_id"] in img_ids and a["category_id"] in cat_ids
    ]
    ann_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_dir / f"instances_{split_name}.json", "w") as fp:
        json.dump(subset_anns, fp)


def main():
    fouc.download_coco_dataset_split(
        str(coco_dir / "validation"),
        "validation",
        classes=classes,
        raw_dir=str(raw_dir),
        scratch_dir=str(scratch_dir),
    )
    print("validation: images downloaded")

    raw_anns = load_raw_anns("val2017")
    cat_id_per_class = {c["name"]: c["id"] for c in raw_anns["categories"] if c["name"] in classes}
    cat_ids = set(cat_id_per_class.values())
    img_ids = set().union(*get_img_ids_per_cat(raw_anns, cat_ids).values())
    write_subset_anns(raw_anns, img_ids, cat_ids, "val2017")
    print("validation: annotations written")

    raw_anns = load_raw_anns("train2017")
    img_ids_per_cat = get_img_ids_per_cat(raw_anns, cat_ids)
    rng = random.Random(seed)
    img_ids = set()
    for class_name in sampled_classes:
        cat_img_ids = sorted(img_ids_per_cat[cat_id_per_class[class_name]])
        img_ids |= set(rng.sample(cat_img_ids, min(n_per_class, len(cat_img_ids))))

    fouc.download_coco_dataset_split(
        str(coco_dir / "train"),
        "train",
        image_ids=sorted(img_ids),
        raw_dir=str(raw_dir),
        scratch_dir=str(scratch_dir),
    )
    print("train: images downloaded")

    write_subset_anns(raw_anns, img_ids, cat_ids, "train2017")
    print("train: annotations written")


if __name__ == "__main__":
    main()
