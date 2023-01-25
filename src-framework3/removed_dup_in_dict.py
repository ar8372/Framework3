from settings import * 


if __name__ == "__main__":
    a = amzcomp1_settings()

    b = list(a.auto_filtered_features.keys())
    
    print(b[::-1])