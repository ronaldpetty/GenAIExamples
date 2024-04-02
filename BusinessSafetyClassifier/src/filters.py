def is_meeting_notes(text):
    indicators = [
        "meeting adjourned",
        "meeting minutes",
        "next meeting"
    ]
    
    for ind in indicators:
        if ind in text.lower():
            return 1 
    return 0


def contain_name(text, analyzer):
    from presidio_analyzer import AnalyzerEngine
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text,
                        entities=["PERSON"],
                        language='en')
    # print(results)
    if len(results)>0:
        return 1
    else:
        return 0


def run_filters(batch_text): 
    labels = []

    for text in batch_text:
        label = 0
        if is_meeting_notes(text) == 1:
            label = 1
        else:
            label = 0
        labels.append(label)
    return labels