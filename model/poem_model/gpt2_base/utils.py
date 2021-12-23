def tag_keyword_and_special_token(df):
    # <e> k1, k2, k3 </e> poem
    keyword_start_marker = "<k>"
    keyword_end_marker = "</k>"
    text = df["text"]
    keyword = df["key_word"]
    tagged_text = keyword_start_marker + keyword + keyword_end_marker + text
    return tagged_text


def get_tagged_data(df):
    data = []

    for i in range(len(df)):
        try:
            data.append(tag_keyword_and_special_token(df.iloc[i]))
        except:
            continue
    return data
