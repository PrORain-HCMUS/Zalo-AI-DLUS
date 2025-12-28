import torch


class DINOv2FeatureExtractor:
    def __init__(self, model_name='facebook/dinov2-small', device='cuda'):
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading DINOv2 model: {model_name}...")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if device == 'cuda' and torch.cuda.is_available():
            self.model.half()
            self.fp16 = True
        else:
            self.fp16 = False

        print(f"DINOv2 loaded successfully on {self.device}")

    def extract_features(self, img_rgb):
        import torch.nn.functional as F
        from PIL import Image
        import cv2

        if isinstance(img_rgb, Image.Image):
            pil_img = img_rgb
        else:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if img_rgb.shape[2] == 3 else img_rgb
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.fp16:
            inputs['pixel_values'] = inputs['pixel_values'].half()

        with torch.no_grad():
            outputs = self.model(**inputs)

        features = outputs.last_hidden_state[:, 0, :]
        features = F.normalize(features, p=2, dim=1)

        return features.squeeze(0)

    def compute_similarity(self, features1, features2):
        return torch.nn.functional.cosine_similarity(
            features1.unsqueeze(0),
            features2.unsqueeze(0),
            dim=1
        ).item()
