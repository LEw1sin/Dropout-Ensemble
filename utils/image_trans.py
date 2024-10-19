class ImageTransform:
    def __init__(self, target_depth=None, target_height=None, target_width=None):
        self.target_depth = target_depth
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, img):
        img = self.normalization(img)
        return img
    
    def normalization(self,img):
        img = (img - img.min()) / (img.max() - img.min())
        return img

    



