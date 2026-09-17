# What the model actually does

Real predictions from `results/`, not illustrations. The task: turn an Egyptian Arabic
voice command into a tool call — one intent plus its slots, as JSON.

## Correct predictions (Qwen3-1.7B + LoRA)

| Spoken Egyptian input | Model output |
|---|---|
| انت بتعرف تطبخ رزبرياني | `{"intent": "cooking_recipe", "slots": {"food_type": "رزبرياني"}}` |
| شغل ببجي | `{"intent": "play_game", "slots": {"game_name": "ببجي"}}` |
| امسح قائمة التسوق بتاعتى | `{"intent": "lists_remove", "slots": {"list_name": "التسوق"}}` |
| اتصل على أقرب مطعم مندي عندهم توصيل | `{"intent": "takeaway_order", "slots": {"business_type": "مطعم", "food_type": "مندي", "order_type": "توصيل"}}` |
| ايه عندي في قائمة المهام بتاعتى | `{"intent": "lists_query", "slots": {"list_name": "المهام"}}` |
| ضيف البروكلي لقائمة البقالة | `{"intent": "lists_createoradd", "slots": {"list_name": "البقالة"}}` |

## Where fine-tuning beats prompting

The same sentences through the fine-tuned 1.7B and through Claude Sonnet 5 with five
examples. Sonnet 5 reads the Arabic correctly; what it misses is the dataset's labelling
convention — it takes greedier spans (`قائمة البقالة` where the label is `البقالة`), adds
slots the schema does not mark, and occasionally picks a defensible but different intent.

| Input | Expected | Qwen3-1.7B + LoRA | Sonnet 5 (5-shot) |
|---|---|---|---|
| اتصل على أقرب مطعم مندي عندهم توصيل | `{"intent": "takeaway_order", "slots": {"business_type": "مطعم", "food_type": "مندي", "order_type": "توصيل"}}` | ✅ `{"intent": "takeaway_order", "slots": {"business_type": "مطعم", "food_type": "مندي", "order_type": "توصيل"}}` | ❌ `{"intent": "recommendation_locations", "slots": {"business_type": "مطعم مندي"}}` |
| ضيف البروكلي لقائمة البقالة | `{"intent": "lists_createoradd", "slots": {"list_name": "البقالة"}}` | ✅ `{"intent": "lists_createoradd", "slots": {"list_name": "البقالة"}}` | ❌ `{"intent": "lists_createoradd", "slots": {"ingredient": "البروكلي", "list_name": "البقالة"}}` |
| احذف الصف الثاني من القائمة | `{"intent": "lists_remove", "slots": {}}` | ✅ `{"intent": "lists_remove", "slots": {}}` | ❌ `{"intent": "lists_remove", "slots": {"list_name": "الصف الثاني"}}` |
| اقرأ قائمة البقالة | `{"intent": "lists_query", "slots": {"list_name": "البقالة"}}` | ✅ `{"intent": "lists_query", "slots": {"list_name": "البقالة"}}` | ❌ `{"intent": "lists_query", "slots": {"list_name": "قائمة البقالة"}}` |

## Where every model still fails

| Input | Expected | Best model's output |
|---|---|---|
| متشغلش عود | `{"intent": "music_dislikeness", "slots": {"music_genre": "عود"}}` | `{"intent": "play_music", "slots": {"player_setting": "عود"}}` |
| قولى تفاصيل الطلبية اللي شريتها | `{"intent": "lists_query", "slots": {}}` | `{"intent": "takeaway_query", "slots": {}}` |
| latest bill simmons podcast | `{"intent": "play_podcasts", "slots": {"person": "bill simmons"}}` | `{"intent": "play_podcasts", "slots": {}}` |

Failures cluster on `music` and `play`: slot values there are song, artist and playlist
names that must be copied character-for-character. See [results.md](results.md) §5.
