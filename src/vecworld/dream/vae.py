"""The panorama VAE, tiled along the wrapping width: SD-Turbo's (dream.painter.TurboPainter) and FLUX's
(flux.painter.FluxPainter) decode and encode a 360-degree latent through it.

Each tile is coded with VAE_MARGIN of wrap-around context, cropped, and crossfaded with its neighbours over their
overlap: per-tile GroupNorm gives each tile its own colour offset, which hard crops showed as vertical lines. The grid
moves with the seed so no boundary stays put.
"""
import numpy as np
from PIL import Image

from vecworld.config import PANO_H, PANO_W, VAE_MARGIN, VAE_STRIDE, VAE_TILE
from vecworld.dream.gpu import check_stop
from vecworld.dream.pano import ramp, tile_origins, wrap_cols


class TiledVae:
    """_decode(lat, shift) -> image and _encode(image, shift) -> latent over self.vae (scaled by self._sf; self._torch,
    self._dev and self._dtype set by the painter). Decodes vae.decode(lat / sf), encodes the latent mean * sf."""

    def _decode(self, lat, shift: int = 0) -> Image.Image:
        torch, w, m = self._torch, lat.shape[-1], VAE_MARGIN
        acc = wsum = None
        wt = torch.from_numpy(ramp(VAE_TILE * 8, (VAE_TILE - VAE_STRIDE) * 8)).to(lat.device)
        for x0 in tile_origins(w, shift):
            check_stop()
            px = self.vae.decode(lat[..., wrap_cols(x0 - m, VAE_TILE + 2 * m, w)] / self._sf).sample
            px = px[..., m * 8:(m + VAE_TILE) * 8].float()
            if acc is None:
                acc = torch.zeros((*px.shape[:-1], w * 8), device=px.device)
                wsum = torch.zeros(w * 8, device=px.device)
            cols = torch.tensor(wrap_cols(x0 * 8, VAE_TILE * 8, w * 8), device=px.device)
            acc.index_add_(-1, cols, px * wt)
            wsum.index_add_(0, cols, wt)
        img = (acc / wsum)[0].clamp(-1, 1)
        return Image.fromarray(((img + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())

    def _encode(self, im: Image.Image, shift: int = 0):
        torch, m = self._torch, VAE_MARGIN
        a = np.asarray(im.convert("RGB").resize((PANO_W, PANO_H)))
        x = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].to(self._dev, self._dtype) / 127.5 - 1
        w = x.shape[-1] // 8
        acc = wsum = None
        wt = torch.from_numpy(ramp(VAE_TILE, VAE_TILE - VAE_STRIDE)).to(x.device)
        for x0 in tile_origins(w, shift):
            check_stop()
            lat = self.vae.encode(x[..., wrap_cols((x0 - m) * 8, (VAE_TILE + 2 * m) * 8, w * 8)]).latent_dist.mean
            lat = lat[..., m:m + VAE_TILE].float()
            if acc is None:
                acc = torch.zeros((*lat.shape[:-1], w), device=x.device)
                wsum = torch.zeros(w, device=x.device)
            cols = torch.tensor(wrap_cols(x0, VAE_TILE, w), device=x.device)
            acc.index_add_(-1, cols, lat * wt)
            wsum.index_add_(0, cols, wt)
        return (acc / wsum * self._sf).to(self._dtype)
