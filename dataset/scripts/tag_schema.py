"""Allowed values for the ChatGPT tags (shared by merge and quality check)."""
TEXT = {
    "genre": {"billionaire_ceo_romance", "mafia_romance", "werewolf_paranormal", "fantasy_cultivation",
              "historical_royal", "revenge_betrayal", "family_drama", "marriage_divorce",
              "hidden_identity_status", "thriller_crime", "comedy", "unclear"},
    "hook_type": {"violence_or_slap", "public_humiliation", "betrayal_reveal", "identity_or_status_reveal",
                  "threat_or_ultimatum", "romantic_or_sexual_shock", "insult_line", "rescue_entrance",
                  "narrator_setup", "question_to_viewer", "calm_dialogue", "no_speech", "unclear"},
    "ending_type": {"unanswered_question", "pending_threat", "interrupted_reveal", "arrival_or_entrance",
                    "decision_pending", "resolved", "call_to_action_download", "no_speech", "unclear"},
    "setting": {"modern", "historical", "fantasy", "unclear"},
    "confidence": {"low", "medium", "high"},
}
TROPES = {"jilted_or_cheated", "contract_or_fake_marriage", "secret_baby_or_pregnancy", "hidden_rich_or_powerful",
          "revenge_comeback", "humiliation_underdog", "rescue_protector", "forbidden_or_age_gap",
          "rebirth_second_chance", "love_triangle", "evil_family_or_inlaws", "amnesia", "enemies_to_lovers", "none"}
IMAGE = {
    "visual_hook": {"violence_slap_fight", "intimacy_kiss", "crying_or_shock_closeup", "wedding_or_formal_event",
                    "wealth_luxury_display", "public_humiliation_crowd", "injury_blood_hospital", "pregnancy_or_baby",
                    "supernatural_or_fantasy", "text_overlay_only", "calm_talking", "other", "unclear"},
    "shot_type": {"closeup", "medium", "wide", "mixed"},
    "people_count": {"0", "1", "2", "3+"},
    "visual_setting": {"home", "office", "street_outdoor", "party_event", "hospital", "historical", "fantasy",
                       "vehicle", "other", "unclear"},
    "confidence": {"low", "medium", "high"},
}
