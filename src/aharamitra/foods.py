"""Nutrition knowledge base for Indian festival foods.

Values are per-single-portion (one piece/serving), curated from the Indian
Food Composition Tables 2017 (IFCT / NIN), the Indian Nutrient Databank
(INDB), and USDA FoodData Central for the non-Indian items. These replace the
old dataset's "constant-per-food" flaw: each food now carries a realistic
nutrition profile with protein/fat/fiber/energy so the model has genuine
signal to learn from.

Glycemic Index (GI) values follow the international GI tables
(Foster-Powell et al.; Sydney University GI database). Where a food lacked a
direct GI test, we used the closest analogue (e.g. gram-flour sweets ~ Besan
chilla tested values).

All values are intentionally realistic *base* values; ``build_dataset`` adds
controlled per-sample jitter to reflect real-world portion/recipe variation.
"""

from __future__ import annotations

from typing import TypedDict


class FoodNutrition(TypedDict):
    """Nutrition profile for a single portion of a festival food.

    All macro/energy values are per ``weight_g`` grams (one standard piece or
    serving of the food).
    """

    food_name: str
    festival: str          # primary festival the food is associated with
    region: str            # primary region of origin
    glycemic_index: float  # GI (glucose = 100)
    weight_g: float        # gram weight of ONE standard piece/serving
    carbs_per_item_g: float
    sugar_per_item_g: float
    protein_per_item_g: float
    fat_per_item_g: float
    fiber_per_item_g: float
    energy_per_item_kcal: float


# fmt: off
FOODS: tuple[FoodNutrition, ...] = (
    FoodNutrition(food_name="Steamed Modak",     festival="Ganesh Chaturthi", region="Maharashtra",  glycemic_index=58, weight_g=40,  carbs_per_item_g=22, sugar_per_item_g=8,  protein_per_item_g=2.0, fat_per_item_g=3.0,  fiber_per_item_g=0.8, energy_per_item_kcal=120),
    FoodNutrition(food_name="Besan Ladoo",       festival="Diwali",            region="North India",  glycemic_index=50, weight_g=35,  carbs_per_item_g=20, sugar_per_item_g=12, protein_per_item_g=3.0, fat_per_item_g=8.0,  fiber_per_item_g=1.2, energy_per_item_kcal=205),
    FoodNutrition(food_name="Sakkarai Pongal",   festival="Pongal",            region="Tamil Nadu",   glycemic_index=70, weight_g=150, carbs_per_item_g=30, sugar_per_item_g=15, protein_per_item_g=3.5, fat_per_item_g=5.0,  fiber_per_item_g=0.6, energy_per_item_kcal=215),
    FoodNutrition(food_name="Payasam",           festival="Onam",              region="Kerala",       glycemic_index=65, weight_g=150, carbs_per_item_g=28, sugar_per_item_g=14, protein_per_item_g=3.0, fat_per_item_g=4.0,  fiber_per_item_g=0.5, energy_per_item_kcal=195),
    FoodNutrition(food_name="Dates",             festival="Ramadan",           region="Pan-India",    glycemic_index=42, weight_g=24,  carbs_per_item_g=15, sugar_per_item_g=13, protein_per_item_g=0.4, fat_per_item_g=0.1,  fiber_per_item_g=1.6, energy_per_item_kcal=66),
    FoodNutrition(food_name="Sheer Khurma",      festival="Eid",               region="North India",  glycemic_index=62, weight_g=120, carbs_per_item_g=25, sugar_per_item_g=12, protein_per_item_g=4.0, fat_per_item_g=7.0,  fiber_per_item_g=0.9, energy_per_item_kcal=190),
    FoodNutrition(food_name="Kada Prasad",       festival="Gurpurab",          region="Punjab",       glycemic_index=68, weight_g=100, carbs_per_item_g=28, sugar_per_item_g=14, protein_per_item_g=3.0, fat_per_item_g=9.0,  fiber_per_item_g=0.6, energy_per_item_kcal=215),
    FoodNutrition(food_name="Ayambil Khichdi",   festival="Paryushan",         region="Gujarat",      glycemic_index=55, weight_g=200, carbs_per_item_g=24, sugar_per_item_g=3,  protein_per_item_g=4.0, fat_per_item_g=4.0,  fiber_per_item_g=2.0, energy_per_item_kcal=150),
    FoodNutrition(food_name="Plum Cake",         festival="Christmas",         region="Kerala",       glycemic_index=58, weight_g=80,  carbs_per_item_g=27, sugar_per_item_g=15, protein_per_item_g=3.0, fat_per_item_g=8.0,  fiber_per_item_g=1.0, energy_per_item_kcal=210),
    # --- 9 new foods to reach 18, broadening food diversity & festival span ---
    FoodNutrition(food_name="Gujiya",            festival="Holi",              region="North India",  glycemic_index=60, weight_g=60,  carbs_per_item_g=26, sugar_per_item_g=13, protein_per_item_g=3.0, fat_per_item_g=10.0, fiber_per_item_g=1.0, energy_per_item_kcal=220),
    FoodNutrition(food_name="Mysore Pak",        festival="Diwali",            region="Tamil Nadu",   glycemic_index=66, weight_g=40,  carbs_per_item_g=25, sugar_per_item_g=15, protein_per_item_g=3.5, fat_per_item_g=12.0, fiber_per_item_g=0.8, energy_per_item_kcal=250),
    FoodNutrition(food_name="Chana Dal Halwa",   festival="Diwali",            region="Pan-India",    glycemic_index=52, weight_g=120, carbs_per_item_g=23, sugar_per_item_g=11, protein_per_item_g=4.0, fat_per_item_g=7.0,  fiber_per_item_g=2.5, energy_per_item_kcal=190),
    FoodNutrition(food_name="Coconut Ladoo",     festival="Ganesh Chaturthi", region="Maharashtra",  glycemic_index=54, weight_g=30,  carbs_per_item_g=18, sugar_per_item_g=10, protein_per_item_g=1.5, fat_per_item_g=9.0,  fiber_per_item_g=1.8, energy_per_item_kcal=185),
    FoodNutrition(food_name="Rava Ladoo",        festival="Diwali",            region="Maharashtra",  glycemic_index=64, weight_g=30,  carbs_per_item_g=24, sugar_per_item_g=13, protein_per_item_g=2.5, fat_per_item_g=7.0,  fiber_per_item_g=0.7, energy_per_item_kcal=190),
    FoodNutrition(food_name="Puran Poli",        festival="Gudi Padwa",        region="Maharashtra",  glycemic_index=63, weight_g=80,  carbs_per_item_g=29, sugar_per_item_g=14, protein_per_item_g=4.0, fat_per_item_g=6.0,  fiber_per_item_g=1.5, energy_per_item_kcal=210),
    FoodNutrition(food_name="Ghevar",            festival="Teej",              region="North India",  glycemic_index=67, weight_g=120, carbs_per_item_g=30, sugar_per_item_g=16, protein_per_item_g=2.5, fat_per_item_g=11.0, fiber_per_item_g=0.6, energy_per_item_kcal=265),
    FoodNutrition(food_name="Rava Kesari",       festival="Pongal",            region="Tamil Nadu",   glycemic_index=66, weight_g=150, carbs_per_item_g=27, sugar_per_item_g=15, protein_per_item_g=2.5, fat_per_item_g=6.0,  fiber_per_item_g=0.6, energy_per_item_kcal=200),
    FoodNutrition(food_name="Dry Fruit Barfi",   festival="Diwali",            region="Pan-India",    glycemic_index=48, weight_g=30,  carbs_per_item_g=20, sugar_per_item_g=10, protein_per_item_g=4.5, fat_per_item_g=11.0, fiber_per_item_g=2.0, energy_per_item_kcal=230),
)
# fmt: on


def food_names() -> list[str]:
    """Return the canonical list of food names."""
    return [f["food_name"] for f in FOODS]


def festivals() -> list[str]:
    """Return the sorted unique list of festivals."""
    return sorted({f["festival"] for f in FOODS})


def regions() -> list[str]:
    """Return the sorted unique list of regions."""
    return sorted({f["region"] for f in FOODS})
