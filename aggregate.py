#!/usr/bin/env python3
"""
Adaptive Hierarchical Aggregation CLI

Aggregates branch profiles into hierarchical timeline:
    Branches → Weeks → Months → Years → Life Profile
"""

import asyncio
import json
import os
from pathlib import Path

from toolkit.aggregator import AdaptiveAggregator


async def main_async():
    """Main async function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive hierarchical aggregation",
        epilog="By default, skips existing period summaries. Use --force to reprocess."
    )
    parser.add_argument(
        "profiles_dir", type=Path, help="Directory containing branch profiles"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for aggregations"
    )
    parser.add_argument(
        "--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--parallel", type=int, default=5, help="Max parallel workers (default: 5)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Reprocess existing summaries (skip by default)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed progress messages"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  No API key provided. Set OPENROUTER_API_KEY or use --api-key")
        print("   Aggregation will run but skip synthesis (metadata-only placeholders)")

    # Load profiles
    profiles = []
    for profile_file in sorted(args.profiles_dir.glob("*.json")):
        with open(profile_file) as f:
            profile = json.load(f)
            profile["_source_file"] = profile_file.name  # Mark as branch profile
            profiles.append(profile)

    if not profiles:
        print(f"❌ No profiles found in {args.profiles_dir}")
        return

    # Create aggregator
    aggregator = AdaptiveAggregator(
        profiles_dir=args.profiles_dir,
        output_dir=args.output_dir,
        api_key=api_key,
        skip_existing=not args.force,
        max_parallel=args.parallel,
        verbose=args.verbose,
    )

    # Build hierarchy
    try:
        life_profile = await aggregator.build_hierarchy(profiles)
        # Aggregator already prints comprehensive completion summary
    except Exception as e:
        print(f"\n❌ Aggregation failed: {e}")

        # Show debug log locations
        debug_dir = args.output_dir.parent / "debug"
        print(f"\n📁 Check debug logs for details:")
        print(f"   Period logs: {debug_dir / 'aggregator' / 'periods'}/")
        print(f"   Life log: {debug_dir / 'aggregator' / 'life'}/")

        raise


def main():
    """Entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
