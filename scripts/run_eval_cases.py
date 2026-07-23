"""Compatibility entry point for the non-circular evaluation capture flow.

New code should invoke ``capture_eval_results.py`` directly. This wrapper is
kept so older commands cannot accidentally auto-label predictions as truth.
"""

from capture_eval_results import main


if __name__ == "__main__":
    main()
