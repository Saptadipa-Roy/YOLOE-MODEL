# # Promt free (YOLOE model)
# model = YOLOE("yoloe-11l-seg-pf.pt")
# results = model.predict( "Room-Video.mp4",
#                          show = True,
#                          save = True,
#                          show_conf = False)

# for r in results: # Extract the results
#     boxes = r.boxes.xyxy.cpu().tolist()
#     cls = r.boxes.cls.cpu().tolist()
#     for b,c in zip(boxes,cls): #display result
#         print(f"box: {b},cls:{model.names[c]}")

from ultralytics import YOLOE
# Text promt (YOLOE model)
model = YOLOE("yoloe-11l-seg.pt")
names = ["camel walking"]
model.set_classes(names,model.get_text_pe(names))
model.predict("Camel Walking.mp4",
              show = True,
              save = True
)