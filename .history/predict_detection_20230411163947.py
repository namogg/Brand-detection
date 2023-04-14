q
#cv2.waitKey()
model_classification = create_model(27,train = False,load_path='classification_model.pt')
print(model_classification)
brand_list = []
for result in results: 
    bbox = result.boxes.xyxy 
    if bbox.numel():
        xmin, ymin, xmax, ymax = bbox[0].int()
        # Crop image using bounding box coordinates
        cropped_img = result.orig_img[ymin:ymax, xmin:xmax, :]
        img = Image.fromarray(cropped_img)
        img_tensor = transform_image(img,train = False)
        brand = show_prediction(img_tensor,model_classification,class_names,show = True,threshold=0.7)
        if brand not in brand_list:
            brand_list.append(brand)

print("Brand detected: "+str(len(brand_list)))
print(brand_list)