from PIL import Image, ImageFilter, ImageDraw

BBOX_COLORS = ["green", "red", "blue", "yellow", "orange", "purple", "white", "brown", "pink"]


def _bbox_union(bounding_boxes):
    x, y, w, h = 999, 999, 0, 0
    for box in bounding_boxes:
        bx, by, bw, bh = box
        if bx < x: x = bx
        if by < y: y = by
        if bw > w: w = bw
        if bh > h: h = bh
    return [x, y, w, h]


def blur_except_boxes(image, bounding_boxes):
    """Apply blur to all areas except the union of bounding boxes."""
    blurred_image = image.filter(ImageFilter.GaussianBlur(15))
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(_bbox_union(bounding_boxes), fill=255)
    return Image.composite(image, blurred_image, mask)


def draw_bounding_boxes(image, bounding_boxes, each_bbox=False, outline="green"):
    draw = ImageDraw.Draw(image)
    if each_bbox:
        for bi, box in enumerate(bounding_boxes):
            draw.rectangle(list(box), outline=BBOX_COLORS[bi % len(BBOX_COLORS)], width=2)
    else:
        draw.rectangle(_bbox_union(bounding_boxes), outline=outline, width=2)
    return image


def crop_box(image, bounding_box):
    if isinstance(bounding_box, list):
        bounding_box = bounding_box[0]
    return image.crop(bounding_box)


def transform_images(image_file, bbox_lst, type="blur", each_bbox=False, max_dim=None, scale=0.8):
    """Load image and apply blur+box draw or crop. Optionally downscale if width > max_dim."""
    image = Image.open(image_file).convert("RGB")
    if type == "blur":
        ret_image = blur_except_boxes(image, bbox_lst)
        ret_image = draw_bounding_boxes(ret_image, bbox_lst, each_bbox=each_bbox)
    elif type == "crop":
        ret_image = crop_box(image, bbox_lst)
    else:
        raise ValueError(f"Unknown transform type: {type}")

    if max_dim and ret_image.size[0] > max_dim:
        new_size = tuple(int(i * scale) for i in ret_image.size)
        ret_image = ret_image.resize(new_size, Image.LANCZOS)
    return ret_image
