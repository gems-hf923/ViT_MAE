import numpy as np 
import torch.nn.functional as F
import torch.nn as nn
import torch

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=(187, 32), patch_size=(17, 4), in_chans=1, embed_dim=768):
        """
        Initializes the PatchEmbed module.

        This method initializes the PatchEmbed module by setting the image size, patch size, grid size, number of patches, input channels, and the projection layer for patch embedding.

        Args:
            img_size (tuple, optional): The size of the input image. Defaults to (187, 32).
            patch_size (tuple, optional): The size of the patch. Defaults to (17, 4).
            in_chans (int, optional): The number of input channels. Defaults to 1.
            embed_dim (int, optional): The embedding dimension. Defaults to 768.
        """
        super().__init__()
        img_size = (img_size if isinstance(img_size, tuple) else (img_size, img_size))
        patch_size = (patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size))
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # (H, W)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.in_chans = in_chans  # Ensure in_chans is properly set
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # Convolution for patch embedding

    def forward(self, x):
        """
        Forward pass through the PatchEmbed module.
        This method performs the forward pass through the PatchEmbed module, which involves embedding the input image into a sequence of patches. It first checks the input dimensions and ensures the input has the correct shape. Then, it applies a convolutional layer to the input image to generate the patch embeddings. The output is a tensor of shape (batch_size, num_patches, embed_dim), where each patch is embedded into a vector of dimension embed_dim.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The embedded patch sequence tensor.
        """
        # Check input dimensions
        if x.ndim == 3:  # If there's no channel dimension, add it
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        if x.ndim != 4:
            raise ValueError(f"Expected input to have 4 dimensions (batch_size, channels, height, width), but got {x.ndim} dimensions.")

        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        
        return x
    
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=True):
    """
    Generate 2D sine-cosine positional embeddings for rectangular patches.
    embed_dim: The total embedding dimension (e.g., 768 for ViT).
    grid_size_h: Number of patches along height.
    grid_size_w: Number of patches along width.
    cls_token: Whether to include the [CLS] token embedding.
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even for sine-cosine embeddings."
    
    # Create grid of shape (2, grid_size_h * grid_size_w) where grid[0] is for height and grid[1] is for width
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # Shape [2, grid_size_h, grid_size_w]
    grid = np.stack(grid, axis=0)  # Shape [2, grid_size_h, grid_size_w]
    grid = grid.reshape([2, grid_size_h * grid_size_w])  # Shape [2, grid_size_h * grid_size_w]

    # Calculate the 2D sine-cosine embedding
    pos_embed = np.zeros((grid.shape[1], embed_dim))
    dim_half = embed_dim // 2

    for i in range(dim_half):
        pos_embed[:, i] = np.sin(grid[0] / 10000 ** (2 * i / embed_dim))  # height-based sine
        pos_embed[:, dim_half + i] = np.cos(grid[1] / 10000 ** (2 * i / embed_dim))  # width-based cosine

    # Add [CLS] token embedding
    if cls_token:
        cls_embed = np.zeros([1, embed_dim])
        pos_embed = np.concatenate([cls_embed, pos_embed], axis=0)

    return pos_embed

class MLP(nn.Module):
    """ Feed-forward layer """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        """
        Initializes the MLP module.

        Args:
            in_features (int): The input dimension.
            hidden_features (int, optional): The dimension of the hidden layer. Defaults to None.
            out_features (int, optional): The output dimension. Defaults to None.
            drop (float, optional): The dropout rate. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass through the MLP module.

        This method applies the feed-forward neural network (FFNN) to the input tensor. It first applies a linear layer, followed by a GELU activation function, and then another linear layer. Dropout is applied after each linear layer to prevent overfitting.

        Args:
            x (torch.Tensor): The input tensor to the MLP module.

        Returns:
            torch.Tensor: The output tensor from the MLP module.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Initializes the Attention module.

        Args:
            dim (int): The input dimension.
            num_heads (int, optional): The number of attention heads. Defaults to 8.
            qkv_bias (bool, optional): If True, adds a learnable bias to the query, key, and value linear layers. Defaults to False.
            attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.
            proj_drop (float, optional): Dropout rate for output projection. Defaults to 0.
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Forward pass through the Attention module.

        This method computes the attention weights and applies them to the input tensor. It first projects the input into query, key, and value tensors. Then, it computes the attention weights using the scaled dot product attention mechanism. The attention weights are then applied to the value tensor to compute the output. Finally, the output is projected back to the original dimensionality and dropout is applied.

        Args:
            x (torch.Tensor): The input tensor to the Attention module.

        Returns:
            torch.Tensor: The output tensor from the Attention module.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)  # split qkv
        
        # Use PyTorch's SDPA for efficient computation
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DropPath(nn.Module):
    """ Drop paths (stochastic depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        """
        Initializes the DropPath module.

        Args:
            drop_prob (float, optional): The probability of dropping a path. Defaults to None.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        Forward pass through the DropPath module.

        This method applies stochastic depth to the input tensor. It randomly drops out entire channels of the input tensor with a probability specified by `drop_prob`. The dropped channels are replaced with zeros. This is done to simulate the effect of randomly dropping out paths during training, which can help improve the robustness of the model.

        Args:
            x (torch.Tensor): The input tensor to the DropPath module.

        Returns:
            torch.Tensor: The output tensor from the DropPath module, with some channels potentially dropped out.
        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Generate a mask with probability keep_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Broadcasting the mask to match input dimensions
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Convert random_tensor to binary mask
        output = x.div(keep_prob) * random_tensor
        return output

class Block(nn.Module):
    """ Transformer Block with SDPA Attention, MLP, and Custom Drop Path """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        """
        Initializes the Block module.

        Args:
            dim (int): The dimension of the input tensor.
            num_heads (int): The number of heads in the multi-head attention.
            mlp_ratio (float, optional): The ratio of the hidden layer size to the input layer size in the MLP. Defaults to 4..
            qkv_bias (bool, optional): Whether to include bias in the qkv projection. Defaults to False.
            drop (float, optional): The dropout rate. Defaults to 0..
            attn_drop (float, optional): The dropout rate for the attention output. Defaults to 0..
            drop_path (float, optional): The dropout rate for the path. Defaults to 0..
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)  # Use SDPA Attention
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # Custom DropPath added here
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        """
        Forward pass through the Block module.

        This method applies the forward pass through the Block module, which consists of layer normalization, attention, drop path, layer normalization again, MLP, and another drop path. The output from the attention module and the MLP module are added to the input after applying the drop path.

        Args:
            x (torch.Tensor): The input tensor to the Block module.

        Returns:
            torch.Tensor: The output tensor from the Block module.
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))  # Apply attention with custom drop path
        x = x + self.drop_path(self.mlp(self.norm2(x)))   # Apply MLP with custom drop path
        return x
    
class VisionTransformerEncoder(nn.Module):
    """ Vision Transformer Encoder with Masked Autoencoder """
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        """
        Initializes the VisionTransformerEncoder module.

        Args:
            embed_dim (int, optional): The dimension of the embedding. Defaults to 768.
            depth (int, optional): The number of transformer blocks. Defaults to 12.
            num_heads (int, optional): The number of attention heads. Defaults to 12.
            mlp_ratio (float, optional): The ratio of the hidden layer size to the input layer size in the MLP. Defaults to 4..
        """
        super().__init__()

        # Remove PatchEmbed here. Only keep the transformer encoder components.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the VisionTransformerEncoder module.

        This method initializes the weights of the VisionTransformerEncoder module by applying a normal distribution to the CLS token.
        """
        # Initialize CLS token
        torch.nn.init.normal_(self.cls_token, std=.02)

    def forward(self, x):
        """
        Forward pass through the VisionTransformerEncoder module.

        This method applies the forward pass through the VisionTransformerEncoder module, which includes prepending the CLS token, passing through transformer blocks, and applying layer normalization. The output is the encoded representation of the input.

        Args:
            x (torch.Tensor): The input tensor to the VisionTransformerEncoder module.

        Returns:
            torch.Tensor: The encoded output tensor from the VisionTransformerEncoder module.
        """
        B = x.shape[0]  # Batch size

        # CLS token embedding (expand the CLS token to match the batch size)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)

        # Concatenate CLS token with input patches
        x = torch.cat((cls_tokens, x), dim=1)

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    
# New MAEViT
class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with Vision Transformer backbone"""
    def __init__(self, img_size=(187, 32), patch_size=(17, 4), in_chans=1,
                 embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        """
        Initializes the Masked Autoencoder with Vision Transformer (ViT) backbone.

        Args:
            img_size (tuple, optional): Size of the input image. Defaults to (187, 32).
            patch_size (tuple, optional): Size of the patch. Defaults to (17, 4).
            in_chans (int, optional): Number of input channels. Defaults to 1.
            embed_dim (int, optional): Dimensionality of the embedding. Defaults to 768.
            depth (int, optional): Number of transformer encoder layers. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            decoder_embed_dim (int, optional): Dimensionality of the decoder embedding. Defaults to 512.
            decoder_depth (int, optional): Number of transformer decoder layers. Defaults to 8.
            decoder_num_heads (int, optional): Number of attention heads in the decoder. Defaults to 16.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()

        # Apply patch embedding only once at the beginning
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Encoder (Transformer) - now it doesn't apply PatchEmbed again
        self.encoder = VisionTransformerEncoder(embed_dim=embed_dim,
                                                depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Projection from encoder embed_dim to decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # Decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans)  # in_chans=1 for grayscale

        self.initialize_weights()

    def initialize_weights(self):
        # Positional embedding initialization
        """
        Initializes the positional embedding for the decoder.
        This method initializes the positional embedding for the decoder by generating 2D sine-cosine embeddings for the patches.
        The embeddings are generated based on the grid size of the patches, which is determined by the image size and patch size.
        """
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.patch_embed.grid_size[0],
                                                    self.patch_embed.grid_size[1],
                                                    cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

    def random_masking(self, x, mask_ratio):
        """
        Randomly masks patches in the input sequence.

        This method randomly selects patches to mask based on the given mask ratio. It generates a noise vector, sorts it to determine the order of patches, and then selects the top patches based on the mask ratio to keep. The rest of the patches are masked. The method returns the masked sequence, the mask itself, and the indices to restore the original order.

        Args:
            x (torch.Tensor): The input sequence of patches.
            mask_ratio (float): The ratio of patches to mask.

        Returns:
            x_masked (torch.Tensor): The input sequence with masked patches.
            mask (torch.Tensor): The binary mask indicating which patches are masked.
            ids_restore (torch.Tensor): The indices to restore the original order of patches.
        """
        N, L, D = x.shape  # batch, num patches, embedding dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # Noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascending order (smallest to largest)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Forward pass through the encoder with optional masking.
        This method performs the forward pass through the encoder, applying patch embedding, encoding, and optional masking of patches based on the given mask ratio.
        It returns the latent representation, the mask, the indices to restore the original order of patches, and the CLS token.

        Args:
            x (torch.Tensor): The input image tensor.
            mask_ratio (float): The ratio of patches to mask. If 0, no masking is applied.

        Returns:
            latent (torch.Tensor): The latent representation of the input image, including the CLS token and masked patches.
            mask (torch.Tensor): The binary mask indicating which patches are masked.
            ids_restore (torch.Tensor): The indices to restore the original order of patches.
            cls_token (torch.Tensor): The CLS token, representing the global image representation.
        """
        # First apply patch embedding (handles the in_chans issue)
        x = self.patch_embed(x)

        # Then pass through encoder
        x = self.encoder(x)

        # Extract the CLS token (first token in the sequence)
        cls_token = x[:, :1, :]  # Shape: (batch_size, 1, embed_dim)

        # Mask the rest of the patches (remove CLS token before masking)
        x_masked, mask, ids_restore = self.random_masking(x[:, 1:], mask_ratio)  # Remove cls token

        # Concatenate the CLS token back to the masked patches
        latent = torch.cat([cls_token, x_masked], dim=1)  # Shape: (batch_size, num_patches + 1, embed_dim)

        return latent, mask, ids_restore, cls_token  # Return the CLS token along with other values

    def forward_decoder(self, x, ids_restore):
        """
        Forward pass through the decoder.
        This method performs the forward pass through the decoder, including the embedding of the input, the application of mask tokens for missing patches, unshuffling to restore the original patch order, and the final prediction.

        Args:
            x (torch.Tensor): The input tensor to the decoder.
            ids_restore (torch.Tensor): The indices to restore the original order of patches.

        Returns:
            x (torch.Tensor): The reconstructed image tensor.
        """
        # Project encoder output to match decoder's embedding dimension
        x = self.decoder_embed(x)

        # Prepare mask tokens for the missing patches
        num_patches_total = ids_restore.shape[1]
        num_patches_visible = x.shape[1] - 1
        mask_tokens = self.mask_token.repeat(x.shape[0], num_patches_total - num_patches_visible, 1)

        # Concatenate mask tokens
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)

        # Unshuffle tokens to restore the original patch order
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # Concatenate the CLS token back
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Pass through decoder transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        
        # Project back to pixel space (grayscale - 1 channel)
        x = self.decoder_pred(x)

        # Unpatchify the output to match the original image size
        x = self.unpatchify(x)

        return x

    def patchify(self, imgs):
        """
        Patchify the input image into smaller patches.

        This method takes an input image and divides it into smaller patches based on the patch size defined in the model. The patches are then reshaped and rearranged to form a tensor that can be processed by the model.

        Args:
            imgs (torch.Tensor): The input image tensor to be patchified.

        Returns:
            x (torch.Tensor): The patchified image tensor.
        """
        p_h, p_w = self.patch_embed.patch_size  # Patch size
        h_patches = imgs.shape[2] // p_h  # Number of patches along height
        w_patches = imgs.shape[3] // p_w  # Number of patches along width

        # Reshape the image into patches
        x = imgs.reshape(shape=(imgs.shape[0], 1, h_patches, p_h, w_patches, p_w)).contiguous()  # Ensure contiguous
        x = torch.einsum('nchpwq->nhwpqc', x).contiguous()  # Rearrange the dimensions and ensure contiguous
        x = x.reshape(shape=(imgs.shape[0], h_patches * w_patches, p_h * p_w * 1)).contiguous()  # Reshape into flattened patches
        return x

    def unpatchify(self, x):
        """
        Unpatchify the input tensor to recover the original image shape.

        This method takes a tensor that has been patchified and reshapes it back into the original image shape. It is the inverse operation of patchify.

        Args:
            x (torch.Tensor): The patchified tensor to be unpatchified.

        Returns:
            img (torch.Tensor): The unpatchified tensor, reshaped to match the original image dimensions.
        """
        x = x[:, 1:, :]  # Remove CLS token
        x = x.contiguous()  # Ensure the tensor is contiguous before further operations

        p_h, p_w = self.patch_embed.patch_size  # Patch height and width (17, 4)
        h_patches = self.patch_embed.img_size[0] // p_h  # 11 patches along height
        w_patches = self.patch_embed.img_size[1] // p_w  # 8 patches along width

        # Reshape and permute to recover the original image shape
        x = x.reshape(shape=(x.shape[0], h_patches, w_patches, p_h, p_w, self.patch_embed.in_chans)).contiguous()  # Ensure contiguous
        x = torch.einsum('nhwpqc->nchpwq', x).contiguous()  # Reorder the dimensions
        img = x.reshape(shape=(x.shape[0], self.patch_embed.in_chans, h_patches * p_h, w_patches * p_w)).contiguous()  # Ensure contiguous

        return img
    
    def forward_loss(self, imgs, pred, mask):
        """
        Calculate the loss between the predicted image and the original image, considering masked patches.

        This method calculates the mean squared error (MSE) loss between the predicted image and the original image, weighted by the mask. The mask indicates which patches are visible and should be considered in the loss calculation.

        Args:
            imgs (torch.Tensor): The original image tensor, shape (batch_size, in_chans, img_height, img_width).
            pred (torch.Tensor): The predicted image tensor, shape (batch_size, in_chans, img_height, img_width).
            mask (torch.Tensor): The mask tensor indicating visible patches, shape (batch_size, num_patches).

        Returns:
            loss (torch.Tensor): The calculated MSE loss, weighted by the mask.
        """
        target = self.patchify(imgs)
        pred_patched = self.patchify(pred)

        # Ensure the shapes are correct before applying the mask
        assert pred_patched.shape == target.shape, f"Prediction and target shapes don't match: {pred_patched.shape} vs {target.shape}"

        # Calculate MSE loss, weighted by the mask
        loss = (pred_patched - target) ** 2
        loss = loss.mean(dim=-1)

        # Normalize by the number of visible patches (use the mask to select visible patches)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, imgs, mask_ratio=0.75):
        """
        Forward pass through the model, including encoding, decoding, and loss calculation.

        This method orchestrates the forward pass through the model, applying masking to the input images, encoding the masked images, decoding the latent representation, and calculating the loss between the original and predicted images.

        Args:
            imgs (torch.Tensor): The input image tensor, shape (batch_size, in_chans, img_height, img_width).
            mask_ratio (float, optional): The ratio of patches to be masked. Defaults to 0.75.

        Returns:
            loss (torch.Tensor): The calculated loss between the original and predicted images.
            pred (torch.Tensor): The predicted image tensor, shape (batch_size, in_chans, img_height, img_width).
            mask (torch.Tensor): The mask tensor indicating visible patches, shape (batch_size, num_patches).
            cls_token (torch.Tensor): The class token tensor, shape (batch_size, embed_dim).
        """
        # Encode the input image (apply masking and return latent representation, mask, ids_restore, and cls_token)
        latent, mask, ids_restore, cls_token = self.forward_encoder(imgs, mask_ratio)

        # Decode the latent representation to get the predicted reconstruction
        pred = self.forward_decoder(latent, ids_restore)

        # Calculate loss
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask, cls_token
    
