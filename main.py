import glob

from src.create_annotations import *

# Label ids of the dataset
category_ids = {
    "outlier": 0,
    "defect": 1,
  
}

# Define which colors match which categories in the images
category_colors = {
   "(0, 0, 0)": 0, # Outlier
    "(255, 255, 255)": 1, # Window
   
}
category_colorss = {
   # "(0, 0, 0)": 0, # Outlier
    "(0, 0, 0)": 0, # Window
   
}
# Define the ids that are a multiplolygon. In our case: wall, roof and sky
multipolygon_ids = [0]

# Get "images" and "annotations" info 
def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    
    for mask_image in glob.glob(maskpath + "*.png"):
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file
        original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpg"

        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = Image.open(mask_image).convert("RGB")
        w, h = mask_image_open.size
        
        # "images" info 
        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask_image_open, w, h)
        for color, sub_mask in sub_masks.items():
            print ("Running on "+maskpath+" ...")
            if category_colors[color] == 1:
                #print ("created!")
                category_id = category_colors[color]
                polygons, segmentations = create_sub_mask_annotation(sub_mask)

                # Check if we have classes that are a multipolygon
                if category_id in multipolygon_ids:
                    # Combine the polygons to calculate the bounding box and area
                    multi_poly = MultiPolygon(polygons)
                                    
                    annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                    annotations.append(annotation)
                    annotation_id += 1
                else:
                    for i in range(len(polygons)):
                        # Cleaner to recalculate this variable
                        segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                        
                        annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                        
                        annotations.append(annotation)
                        annotation_id += 1
            else:
                #print ("skip")
                continue

            # "annotations" info
            
        image_id += 1
    return images, annotations, annotation_id

if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()
    
    for keyword in ["train", "val"]:
        mask_path = "D:/yola/ICS/{}_mask_mono/".format(keyword)
        
        # Create category section
        coco_format["categories"] = create_category_annotation(category_ids)
    
        # Create images and annotations sections
        coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

        with open("D:/yola/ICS/output_json/{}.json".format(keyword),"w") as outfile:
            json.dump(coco_format, outfile)
        
        print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))