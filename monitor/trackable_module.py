from torch import nn


class TrackableModule(nn.Module):
    def get_metrics(self):
        return {}
