import torch
import torch.nn.functional as F
from gradcam.utils import load_images, save_gradcam
from cuda import enable_cuda


class GradCAM:

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    # One hot vector to give as gradient while backpropagating on logits
    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError(f"Invalid layer name: {target_layer}")

    def forward(self, image, raw_image):
        self.image_shape = raw_image.shape[:-1]
        self.logits = self.model(image)
        # print(self.logits)
        self.probs = F.softmax(self.logits, dim=1)
        # print(self.probs)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation pass
        """
        # print('IDS : ', ids)
        one_hot = self._encode_one_hot(ids)
        # print(one_hot)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        # print(gcam.shape)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        # print(B, C, H, W)
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


def plot_gradcam(image_path, model, model_path, layer, classes=None, class_id=None, **kwargs):
    device = enable_cuda()

    image, original_image = load_images(image_path, **kwargs)
    image = torch.stack(image).to(device)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    gcam = GradCAM(model=model, candidate_layers=layer)
    probs, ids = gcam.forward(image, original_image[0])

    if class_id and classes:
        assert type(class_id) is int, "class id value must be an integer."
        if class_id in range(len(classes)):
            ids = torch.tensor([[class_id]], dtype=torch.int64).to(device)
            probs = torch.tensor([[probs[0, class_id]]])
        else:
            raise ValueError('class id is not present in classes.')

    gcam.backward(ids=ids[:, [0]])
    region = gcam.generate(target_layer=layer)

    print(f" #GRADCAM: {classes[ids[0, 0]] if classes else ids[0, 0]} (prob : {probs[0, 0]})")

    # Grad-CAM save
    save_gradcam(
        filename=f"gradcam-{model.__class__.__name__}-{layer}-{classes[ids[0, 0]] if classes else ids[0, 0]}.png",
        gcam=region[0, 0],
        raw_image=original_image[0]
    )

    gcam.remove_hook()
