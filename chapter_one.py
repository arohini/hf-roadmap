from transformers import pipeline


classifier = pipeline("sentiment-analysis")
sentiment_result = classifier(["I've been waiting for a HuggingFace course my whole life.",
                     "While I'm working always getting distracted",
                     "while using pomodoro technique I'm much more productive",
                     "If I mindfully talk I could see that able to build the relationship"])


print(sentiment_result)

generator = pipeline("text-generation", model="distilgpt2")
auto_gen_res = generator(
    "Enjoy spirituality and",
    truncation=True,
    max_new_tokens=256,
    max_length=30,
    num_return_sequences=5,
)
print(auto_gen_res)


classifier = pipeline("zero-shot-classification")
text_classifier = classifier(
    "Consumer electronics giant Apple is expected to announce a $500 million investment in MP Materials, "
    "operator of the only U.S. rare earths mine, Fox Business reported on Tuesday, citing sources familiar "
    "with the deal.",
    candidate_labels=["education", "politics", "business"],
)

print(text_classifier)


unmasker = pipeline("fill-mask")
filled_in = unmasker("He is the best indian cricket player and got world cup for indian cricket team and his name "
                     "is <mask>"
                     "And he is based out of <mask>", top_k=2)
print(filled_in)



question_answerer = pipeline("question-answering")
qna_res = question_answerer(
    question="Who is Supreme Divine?",
    context="Krish can handle anything in this universe, "
            "this source can do any magic has that much power"
            "very kind, protective and playful",
)
print(qna_res)


summarizer = pipeline("summarization")
text_summ = summarizer("""Cycling, in its most familiar form, dates back to at least the 19th century. One example of an early bicycle was 
known as the “hobby horse”, and it later became the “Dandy horse” and then the “accelerator”. 
Early cycling was reserved for the upper-classes and was seen as highly fashionable and decorous – particularly for men.

Women’s cycling, on the other hand, was viewed as trivial and unbecoming. When women were portrayed cycling, 
they were often eroticised and undressed.

The early development of women’s bicycles and cycle-wear was impeded by debates on women’s morality and 
sexual innocence. The bicycle was said to cause “bicycle face” (a face of muscular tension), 
harm reproductive organs and diminish what supposedly little energy women had.

Cycling women were viewed as sexually promiscuous both for the “unnatural” straddling of the bicycle 
and for the freedom cycling offered them. Where were they all cycling to, men wondered.

The development in 1885 of the Rover “safety bicycle” revolutionised women’s cycling. 
It featured a lower mounting position and inspired somewhat of a cycling craze. 
By the 1890s, several million women around the world were cycling.

The influx of female cyclists on the streets created a moral panic for the Victorians. 
The image of the cycling woman came to represent a new type of woman with feminist ambition. 
This led to a discourse known simply as the “woman question”."""
)

print(text_summ)


# This needs model owner permission
# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
#
# translated_context = translator("Ce cours est produit par Hugging Face.")
# print(translated_context)