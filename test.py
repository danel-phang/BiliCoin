from bilicoin import BiliCoin
import cv2, json

# 创建BiliCoin类的实例
model = BiliCoin("models\\bili_coin.onnx", cpu=True)

output_image, class_info = model.detect("images\\test.png", 0.5, 0.45)

cv2.imwrite('images\\output.jpg', output_image)
print(class_info)

# 类别:
    # 0: 'like',       # 点赞
    # 1: 'unlike',     # 未点赞
    # 2: 'dislike',    # 点踩
    # 3: 'undislike',  # 未点踩
    # 4: 'coin',       # 投币
    # 5: 'uncoin',     # 未投币
    # 6: 'bookmarke',  # 收藏
    # 7: 'unbookmarke' # 未收藏
