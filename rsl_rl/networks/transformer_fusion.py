import torch
import torch.nn as nn

class TransformerFusionActor(nn.Module):
    """使用Transformer Encoder融合本体信息和视觉特征的Actor网络
    
    该网络将本体观测（关节状态、速度等）和视觉特征（来自R(2+1)D等）通过
    Transformer进行交互融合，然后通过MLP映射到动作空间。
    
    Args:
        proprioception_dim: 本体感觉输入维度
        vision_feature_dim: 视觉特征维度
        num_actions: 动作维度
        hidden_dim: Transformer的隐藏维度
        num_heads: 多头注意力的头数
        num_layers: Transformer Encoder层数
        mlp_hidden_dims: MLP隐藏层维度
        dropout: Dropout比率
        activation: 激活函数类型
        use_proprio_embedding: 是否对本体信息使用embedding层
        use_vision_embedding: 是否对视觉特征使用embedding层
    """
    
    def __init__(
        self,
        proprioception_dim: int,
        vision_feature_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_hidden_dims: list[int] = [256, 128],
        dropout: float = 0.1,
        activation: str = "relu",
        use_proprio_embedding: bool = True,
        use_vision_embedding: bool = True,
    ):
        super().__init__()
        
        self.proprioception_dim = proprioception_dim
        self.vision_feature_dim = vision_feature_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # 特征嵌入层（将不同模态映射到统一的hidden_dim）
        if use_proprio_embedding:
            self.proprio_embedding = nn.Sequential(
                nn.Linear(proprioception_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            assert proprioception_dim == hidden_dim, \
                "If not using embedding, proprioception_dim must equal hidden_dim"
            self.proprio_embedding = nn.Identity()
        
        if use_vision_embedding:
            self.vision_embedding = nn.Sequential(
                nn.Linear(vision_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            assert vision_feature_dim == hidden_dim, \
                "If not using embedding, vision_feature_dim must equal hidden_dim"
            self.vision_embedding = nn.Identity()
        
        # 位置编码（可学习）
        # Token 0: 本体信息, Token 1: 视觉特征
        self.positional_encoding = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # 输入格式: [batch, seq_len, hidden_dim]
            norm_first=True,   # Pre-LN结构，更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        
        # MLP Head - 将融合后的特征映射到动作
        mlp_dims = [hidden_dim * 2] + list(mlp_hidden_dims) + [num_actions]
        mlp_layers = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:  # 最后一层不加激活函数
                mlp_layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    mlp_layers.append(nn.Dropout(dropout))
        
        self.action_head = nn.Sequential(*mlp_layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        proprioception: torch.Tensor, 
        vision_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            proprioception: 本体感觉输入 [batch, proprioception_dim]
            vision_features: 视觉特征 [batch, vision_feature_dim]
        
        Returns:
            动作输出 [batch, num_actions]
        """
        batch_size = proprioception.shape[0]
        
        # 1. 特征嵌入
        proprio_embedded = self.proprio_embedding(proprioception)  # [batch, hidden_dim]
        vision_embedded = self.vision_embedding(vision_features)    # [batch, hidden_dim]
        
        # 2. 构造序列 [batch, 2, hidden_dim]
        # Token 0: 本体信息, Token 1: 视觉特征
        token_sequence = torch.stack([proprio_embedded, vision_embedded], dim=1)
        
        # 3. 添加位置编码
        token_sequence = token_sequence + self.positional_encoding
        
        # 4. Transformer编码
        # 输出: [batch, 2, hidden_dim]
        encoded_sequence = self.transformer_encoder(token_sequence)
        
        # 5. 融合策略：拼接所有token
        # [batch, 2, hidden_dim] -> [batch, hidden_dim * 2]
        fused_features = encoded_sequence.reshape(batch_size, -1)
        
        # 6. MLP映射到动作
        actions = self.action_head(fused_features)
        
        return actions


def create_transformer_fusion_actor(
    proprioception_dim: int,
    vision_feature_dim: int,
    num_actions: int,
    hidden_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    mlp_hidden_dims: list[int] = None,
    dropout: float = 0.1,
    activation: str = "relu",
    use_proprio_embedding: bool = True,
    use_vision_embedding: bool = True,
) -> TransformerFusionActor:
    """创建TransformerFusionActor的便捷函数
    
    Args:
        proprioception_dim: 本体感觉维度（关节位置、速度、IMU等）
        vision_feature_dim: 视觉特征维度（如R(2+1)D输出的64维）
        num_actions: 动作维度（如12关节机器人的12个力矩）
        hidden_dim: Transformer隐藏维度
        num_heads: 注意力头数（建议hidden_dim能被num_heads整除）
        num_layers: Transformer层数
        mlp_hidden_dims: MLP隐藏层维度列表
        dropout: Dropout比率
        activation: 激活函数 ('relu', 'gelu', 'silu')
        use_proprio_embedding: 是否使用本体信息embedding
        use_vision_embedding: 是否使用视觉特征embedding
    
    Returns:
        TransformerFusionActor实例
    
    Example:
        >>> # 创建融合网络
        >>> actor = create_transformer_fusion_actor(
        ...     proprioception_dim=48,  # 关节状态等
        ...     vision_feature_dim=64,   # R(2+1)D输出
        ...     num_actions=12,          # 12个关节力矩
        ...     hidden_dim=256,
        ...     num_heads=4,
        ...     num_layers=2,
        ... )
        >>> 
        >>> # 前向传播
        >>> proprio = torch.randn(32, 48)
        >>> vision = torch.randn(32, 64)
        >>> actions = actor(proprio, vision)
        >>> print(actions.shape)  # [32, 12]
    """
    if mlp_hidden_dims is None:
        mlp_hidden_dims = [256, 128]
    
    # 验证参数
    assert hidden_dim % num_heads == 0, \
        f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
    
    model = TransformerFusionActor(
        proprioception_dim=proprioception_dim,
        vision_feature_dim=vision_feature_dim,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=dropout,
        activation=activation,
        use_proprio_embedding=use_proprio_embedding,
        use_vision_embedding=use_vision_embedding,
    )
    
    # 打印网络信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Created Transformer Fusion Actor:")
    print(f"{'='*60}")
    print(f"Input:")
    print(f"  - Proprioception: [{proprioception_dim}]")
    print(f"  - Vision Features: [{vision_feature_dim}]")
    print(f"\nArchitecture:")
    print(f"  - Hidden Dim: {hidden_dim}")
    print(f"  - Transformer Layers: {num_layers}")
    print(f"  - Attention Heads: {num_heads}")
    print(f"  - MLP Hidden Dims: {mlp_hidden_dims}")
    print(f"  - Dropout: {dropout}")
    print(f"\nOutput:")
    print(f"  - Actions: [{num_actions}]")
    print(f"\nParameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    print(f"{'='*60}\n")
    
    return model
