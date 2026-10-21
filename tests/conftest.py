"""Keep tests away from the real home directory: no automatic skill installation into ~/.claude or ~/.codex."""
import os

os.environ["KJOB_NO_SKILLS"] = "1"
