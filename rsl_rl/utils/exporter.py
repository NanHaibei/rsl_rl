import os

def export_policy_as_onnx(
    policy: object, path: str, policy_type: str = None, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        path: The path to the saving directory.
        policy_type: The policy type (deprecated, now auto-detected from policy class).
        normalizer: The empirical normalizer module. If None, Identity is used.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    # 如果提供了policy_type，保持向后兼容性
    if policy_type is not None:
        print(f"Warning: policy_type parameter is deprecated. The policy type is now auto-detected from the policy class.")
    
    # 使用策略对象自带的export_to_onnx方法
    if hasattr(policy, 'export_to_onnx'):
        policy.export_to_onnx(path, filename, normalizer, verbose)
    else:
        raise ValueError(f"Policy {policy.__class__.__name__} does not have export_to_onnx method. All actor-critic classes should now implement their own export_to_onnx method.")
