import math
import torch
import torch.nn as nn
from src.utils import goodness_functions, get_accuracy

class ReLU_full_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class FF_model(torch.nn.Module):
    def __init__(self, opt):
        super(FF_model, self).__init__()
        self.opt = opt
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers
        self.act_fn = ReLU_full_grad

        self.model = nn.ModuleList([nn.Linear(self.opt.input.num_features, self.num_channels[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))

        self.ff_loss = nn.BCEWithLogitsLoss()
        self.threshold = None

        self.running_means = [
            torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5
            for i in range(self.opt.model.num_layers)
        ]

        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(self.opt.model.num_layers - 1)
        )
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, 10, bias=False)
        )
        self.classification_loss = nn.CrossEntropyLoss()
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0]))
                torch.nn.init.zeros_(m.bias)
        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _calc_peer_normalization_loss(self, idx, z):
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0)
        self.running_means[idx] = self.running_means[idx].detach() * self.opt.model.momentum + mean_activity * (1 - self.opt.model.momentum)
        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        goodness_type = getattr(self.opt.model, 'goodness_type', 'sum_of_squares')
        goodness_fn = goodness_functions.get(goodness_type, goodness_functions['sum_of_squares'])
        goodness = goodness_fn(z)

        if 'mean' in goodness_type:
            default_threshold = 2.0
        else:
            default_threshold = z.shape[1]

        current_threshold = getattr(self, 'threshold', default_threshold)
        logits = goodness - current_threshold
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (torch.sum((torch.sigmoid(logits) > 0.5) == labels) / z.shape[0]).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.input.batch_size] = 1

        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn.apply(z)

            if self.opt.model.peer_normalization > 0:
                peer_loss = self._calc_peer_normalization_loss(idx, z)
                scalar_outputs["Peer Normalization"] += peer_loss
                scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            z = z.detach()
            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(inputs, labels, scalar_outputs=scalar_outputs)
        return scalar_outputs

    def forward_downstream_classification_model(self, inputs, labels, scalar_outputs=None):
        if scalar_outputs is None:
            scalar_outputs = {"Loss": torch.zeros(1, device=self.opt.device)}

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)
        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                z = self._layer_norm(z)
                if idx >= 1:
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim=-1)
        output = self.linear_classifier(input_classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = get_accuracy(self.opt, output.data, labels["class_labels"])

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs

def get_model_and_optimizer(opt):
    model = FF_model(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    main_model_params = [
        p for p in model.parameters()
        if all(p is not x for x in model.linear_classifier.parameters())
    ]
    optimizer = torch.optim.SGD([
        {
            "params": main_model_params,
            "lr": opt.training.learning_rate,
            "weight_decay": opt.training.weight_decay,
            "momentum": opt.training.momentum,
        },
        {
            "params": model.linear_classifier.parameters(),
            "lr": opt.training.downstream_learning_rate,
            "weight_decay": opt.training.downstream_weight_decay,
            "momentum": opt.training.momentum,
        },
    ])
    return model, optimizer
