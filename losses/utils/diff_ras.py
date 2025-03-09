import torch
from torch import nn
from skimage.draw import polygon


class DifferentiableRasterization(nn.Module):
    """Transforms vectorized polygons into a rasterized mask in a differentiable way.

    ...

    Attributes
    ----------
    size: int
        the size of the output image
    tau: float
        softness tau controls the smoothness of the rasterized values at the polygon boundary areas.

    Methods
    -------
    forward(polygons)
        forward function that turns the vectorized elements into a mask.
        vertices should have shape (batch, n_polygons, k_vertices, 2)
    """

    def __init__(self, size, device, tau=2.):
        super().__init__()
        self.device = device
        self.size = size
        self.tau = tau
        self.mesh = (
            torch.stack(
                    torch.meshgrid(torch.arange(self.size), torch.arange(self.size)),
                    dim=-1,
                ).reshape(-1, 2)
        ).view(1, self.size*self.size, 1, 2).to(device)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def D(diff):
        """Returns the summed, batched, flattened distance matrix from any point in the output plane to the polygons

        Parameters
        ----------
        diff: torch.Tensor
        difference between each point in the output plane to the polygons

        """
        return torch.linalg.norm(diff, dim=-1).sum(dim=-1)

    def C(self, polygons):
        polygons = polygons[0].detach().to('cpu').numpy()
        n, k, _ = polygons.shape

        canvas = torch.full((n,self.size,self.size), -1)

        for index in range(n):
            vertices = polygons[index]
            rr, cc = polygon(vertices[:, 0], vertices[:, 1], (self.size, self.size))
            canvas[index, rr, cc] = 1

        return canvas.to(self.device)

    def forward(self, polygons):
        """forward function that turns the vectorized elements into a mask."""
        # Convert Polygons from (x,y) to (y,x) format for image data
        polygons = polygons.roll(1, dims=-1) * self.size

        # Resize polygons and mesh to the same size
        b, n, k, _ = polygons.shape
        polygons = polygons.reshape(-1, 1, k, 2)
        mesh = self.mesh.repeat(b * n, 1, 1, 1)

        # Calculate difference between polygon points and mesh
        diff = polygons - mesh

        # Calculate {-1,1} for in/out of polygon (C) and summed distance (D)
        C = self.C(polygons).reshape(b, n, self.size, self.size)
        D = self.D(diff).reshape(b, n, self.size, self.size)

        # Put everything together
        result = self.sigmoid(C * D / (self.tau * self.size))

        return result