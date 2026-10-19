"""Slim application entry with early offline helper dispatch."""
import sys

def main():
    if sys.argv[1:2] in (["--qlh-cut-layers"], ["--qlh-convert-hf-to-gguf"]):
        from model_supply_runtime import dispatch_helper
        return dispatch_helper(sys.argv[1], sys.argv[2:])
    from app_maintenance import main as maintenance
    if sys.argv[1:2] == ["--maintenance"]:
        return maintenance(sys.argv[2:])
    return maintenance(["start", *sys.argv[1:]])

if __name__ == "__main__":
    raise SystemExit(main())
