import cv2

bardet = cv2.barcode_BarcodeDetector()
img = cv2.imread("data/barcode/barcodes-on-a-cookie-pack.jpg")
ok, decoded_info, decoded_type, corners = bardet.detectAndDecode(img)

for info in decoded_info:
    print(info)

img = cv2.resize(img, (600, 400))
cv2.imshow("1", img)
cv2.waitKey()
