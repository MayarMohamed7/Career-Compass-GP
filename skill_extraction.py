import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load the skill dictionary (Adjust the path as necessary)
skill_df = pd.read_csv("/content/drive/MyDrive/dataset_grad/first_trans_try.csv")
skill_keywords = skill_df["Skills"].astype(str).str.lower().tolist()
ignore_words = set(["me", "and", "or", "i", "myself", "experience", "excellent", "skill", "strong", "good", "be", "using","use", "skills"])
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(text) for text in skill_keywords if text not in ignore_words]
matcher.add("SKILL_PATTERNS", patterns)


def extract_entities_skills_and_bigrams(text):
    doc = nlp(text)
    probable_skills = set()

   #phrasematcher 3shan l multi word skill
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        span_text = span.text.lower().replace(" ", "_").replace(".", "_")
        probable_skills.add(span_text)

    # NER and ngrams
    for ent in doc.ents:
      ent_text = ent.text.lower().replace(" ", "_").replace(".", "_")
      if ent_text not in ignore_words and ent_text in skill_keywords:
        probable_skills.add(ent_text)
    for token1, token2 in zip(doc[:-1], doc[1:]):
        bigram_text = f"{token1.text.lower()} {token2.text.lower()}"
        bigram_key = bigram_text.replace(" ", "_").replace(".", "_")
        if bigram_text in skill_keywords and bigram_key not in ignore_words:
            probable_skills.add(bigram_key)

       # Use PoS tagging to identify nouns and verbs as probable skills
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB']:
            token_text = token.lemma_.lower().replace(" ", "_").replace(".", "_")
            if token_text not in ignore_words and token_text in skill_keywords and token_text not in probable_skills:
                probable_skills.append(token_text)
    
    #n filter l hagat ele tl3t ba 
    final_skills = set()
    for skill in probable_skills:
      if any(skill in multi_word_skill for multi_word_skill in probable_skills if multi_word_skill != skill):
        continue #y3ny lw l'a l skill de mwgoda f mukti word yskipha w myhothash tany
      final_skills.add(skill)
    return list(final_skills)



# Example usage
user_paragraph = "My skills are java, python, C++, machine learning and node js"
probable_skills = extract_entities_skills_and_bigrams(user_paragraph)

print ("Probable Skills:", probable_skills)



