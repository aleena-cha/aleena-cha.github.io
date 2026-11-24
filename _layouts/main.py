"""
2D Farming Simulation Game - Main Entry Point
Location: D:\\P4\\Python\\pygame\\main.py
Upgraded with:
  1. Cute Pastel Visuals, Kawaii Chibi Sprites, & Rounded Pill UI
  2. In-Game Buy & Sell Berry & Seed Shop System (Seeds, Upgrades, & Selling Harvest/Loot)
  3. 'E' Key Control Removed — Uses Spacebar, F, B, C, Enter, and Mouse Clicks
"""
import os
import sys

# Ensure current script directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from game import Game

def main():
    print("=" * 64)
    print("   MEADOW VALLEY - CUTE 2D FARMING & MARKET SIMULATION GAME")
    print("=" * 64)
    print(f"Game Directory: {SCRIPT_DIR}")
    print("Upgraded Features:")
    print("  [*] Cute Pastel Visuals, Kawaii Chibi Farmers, & Blossom Map")
    print("  [*] Buy & Sell Market Shop (Buy Seeds/Upgrades & Sell Harvest)")
    print("  [*] Autonomous Cows & Chickens (Drop Milk & Eggs when Petted)")
    print("  [*] Cozy Barn Storage & Save/Load System")
    print("  [*] Updated Controls ('E' key removed):")
    print("      - [W,A,S,D / Arrows] : Move Farmer")
    print("      - [SPACE / Click]    : Till / Plant / Water / Harvest / Pet")
    print("      - [F / ENTER]        : Open Barn Storage or Market Shop")
    print("      - [B]                : Open Buy & Sell Shop Anytime")
    print("      - [C]                : Open Barn Storage Anytime")
    print("      - [1-6 / Scroll]     : Select Hotbar Slot")
    print("=" * 64)

    try:
        game = Game()
        game.run()
    except KeyboardInterrupt:
        print("\nGame exited by user.")
    except Exception as e:
        print(f"\n[Error] Game encountered an error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
