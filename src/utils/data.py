__all__ = ['get_rows_with_null_authors', 'count_nan_by_column', 'get_distinct_values', 'split_full_authors',
           'get_authors_with_paper_count', 'get_distinct_column_values', 'get_keywords_with_counts',
           'get_rows_with_author_id', 'transform_acm_to_scopus', 'transform_ieee_to_scopus', 'clean_citations',
           'flatten_if_needed', 'flatten_if_multiple']

import itertools

import numpy as np
from pyLDAvis import display

from .scopus import scopus_cols


def get_rows_with_null_authors(frame, column_to_check='Authors', columns_to_return=['Authors', 'Title']):
    if column_to_check not in frame.columns:
        raise ValueError(f"Column {column_to_check} not in frame, choose one of {frame.columns}")
    for column in columns_to_return:
        if column not in frame.columns:
            raise ValueError(f"Column {column} not in frame, choose one of {frame.columns}")
    return frame[frame[column_to_check].isnull()]


def count_nan_by_column(frame):
    nan_counts = frame.isna().sum()
    nan_counts.name = 'NaN'
    nan_frame = nan_counts.to_frame()
    nan_frame['Value'] = len(frame) - nan_frame['NaN']
    nan_frame['Percentage not NaN'] = (len(frame) - nan_frame['NaN']) / len(frame) * 100
    levels = [100, 80, 60, 40, 20, 0]

    for i, level in enumerate(levels[:-1]):
        next_level = levels[i + 1]
        filtered = nan_frame[
            (nan_frame['Percentage not NaN'] <= level) & (nan_frame['Percentage not NaN'] >= next_level)].sort_values(
            'Percentage not NaN', ascending=False)

        if not filtered.empty:
            print(f"Columns with {level}% or more but less than {next_level}% not NaN values\n")
            print(filtered)


def get_distinct_values(frame, column):
    return frame[column].unique()


def split_full_authors(input_string):
    parts = input_string.split('(')

    # Step 2: Extract last name and the rest
    number = parts[1].split(')')[0]
    rest = parts[0].split(', ') if ', ' in parts[0] else [parts[0], '']
    result = [rest[0], rest[1], number]
    return result


def get_authors_with_paper_count(frame):
    frame = frame[filtered_df['Author full names'].notna()].filter(['Author full names']).map(
        lambda x: x.split(';')).explode('Author full names')
    frame = frame.groupby('Author full names').value_counts().reset_index()
    frame.columns = ['Author full names', 'Count']
    frame[['LastName', 'FirstName', 'Id']] = frame['Author full names'].str.extract(
        r'(?P<LastName>.+), (?P<FirstName>.+) \((?P<Id>.+)\)', expand=True)
    frame.drop(columns=['Author full names'], inplace=True)
    return frame.sort_values('Count', ascending=False)


def get_distinct_column_values(frame, column):
    frame = frame.groupby(column).value_counts().reset_index()
    frame.columns = ['Keywords', 'Count']
    return frame.sort_values('Count', ascending=False).reset_index().drop(columns=['index'])


def get_keywords_with_counts(frame, column):
    exploded_keywords = frame[frame[column].notna()].filter([column]).map(lambda x: x.lower().split(';')).explode(
        column)
    exploded_keywords[column] = exploded_keywords[column].str.strip()
    exploded_keywords = exploded_keywords.groupby(column).value_counts().reset_index()
    exploded_keywords.columns = ['Keywords', 'Count']
    return exploded_keywords.sort_values('Count', ascending=False).reset_index().drop(columns=['index'])


def get_author_keywords_with_count(frame):
    return get_keywords_with_counts(frame, 'Author Keywords')


def get_rows_with_author_id(frame, author_id,
                            selected_columns=['Author full names', 'Title', 'Year', 'Source title', 'DOI',
                                              'Index Keywords', 'Author Keywords']):
    return frame[frame['Author full names'].str.contains(author_id, na=False)].filter(selected_columns)


def transform_ieee_to_scopus(frame):
    frame['Index Keywords'] = frame['IEEE Terms']
    frame['Title'] = frame['Document Title']
    frame['Language of Original Document'] = 'English'
    frame['Source'] = 'IEEE'
    frame['Source title'] = frame['Publication Title']
    frame['Year'] = frame['Publication Year']
    frame['Cited by'] = frame['Article Citation Count']
    frame['Document Type'] = frame['Document Identifier']
    frame['Author full names'] = frame['Authors']
    frame['ISBN'] = frame['ISBNs']
    frame['Conference name'] = ''
    frame['Conference code'] = ''
    frame['ieee_file'] = frame['PDF Link']
    frame.drop(columns='IEEE Terms', inplace=True)
    frame.drop(columns='Document Title', inplace=True)
    frame.drop(columns='Publication Year', inplace=True)
    frame.drop(columns='Publication Title', inplace=True)
    frame.drop(columns='Article Citation Count', inplace=True)
    frame.drop(columns='Document Identifier', inplace=True)
    frame.drop(columns='Authors', inplace=True)
    frame.drop(columns='ISBNs', inplace=True)
    return frame.filter(scopus_cols)


def transform_acm_to_scopus(frame):
    frame['Language of Original Document'] = 'English'
    frame['DOI'] = frame['doi']
    frame['Source'] = 'ACM Digital Library'
    frame['Title'] = frame['title']
    frame['Year'] = frame['year']
    frame['Source title'] = frame['booktitle']
    frame['Cited by'] = ''
    frame['Document Type'] = frame['ENTRYTYPE']
    frame['Abstract'] = frame['abstract']
    frame['Author full names'] = frame['author']
    frame['Index Keywords'] = ''
    frame['Author Keywords'] = frame['keywords']
    frame['Publisher'] = ''
    frame['Volume'] = frame['volume']
    frame['Conference name'] = frame['booktitle']
    frame['Conference code'] = ''
    frame['ISBN'] = frame['isbn']
    frame['acm_file'] = frame['ID'].map(lambda x: f"https://dl.acm.org/doi/pdf/{x}")
    frame.drop(columns='doi', inplace=True)
    frame.drop(columns='title', inplace=True)
    frame.drop(columns='booktitle', inplace=True)
    frame.drop(columns='ENTRYTYPE', inplace=True)
    frame.drop(columns='abstract', inplace=True)
    frame.drop(columns='keywords', inplace=True)
    frame.drop(columns='volume', inplace=True)
    # frame.drop(columns='booktitle', inplace=True)
    return frame.filter(scopus_cols)


def clean_citations(series):
    return series.apply(lambda x: 0 if pd.isna(x) or str(x).lower() == 'nan'
    else int(float(x)) if str(x).replace('.', '', 1).isdigit()
    else 0)  # Sum all cleaned values


# Custom aggregation function to flatten only if multiple arrays exist
def flatten_if_needed(series):
    lists = [x.split(';') if isinstance(x, str) else x for x in series.dropna()]  # Remove NaN & ensure lists
    if len(lists) > 1:
        return list(itertools.chain.from_iterable(lists))  # Flatten if multiple
    return lists[0] if lists else []  # Keep single list as is


# Function to flatten lists only if multiple exist
def flatten_if_multiple(series):
    #if series.size > 1:
    #series.describe()
    #print(series.name, series.dropna().values)
    v = series.dropna().values
    print(v, series)
    if v.size == 0:
        return np.nan
    else:
        temp_v = []
        for i in v:
            if type(i) == list:
                temp_v += i
            else:
                temp_v.append(i)
        v = np.array(temp_v)
    print(series.name)
    if series.name == "Author full names":
        display(series.name, v)
    if not series.name in ["Abstract", "Source title", "Language of Original Document", "Title", "Document Type"]:
        #if v.size > 1:
            #print(v, v.size)
        if v.size == 1:
            return v[0]
        else:
            # print(v)
            # print(type(v))
            # print(type(set(v)))
            l  = list(set(v))
            return l if len(l) > 1 else l[0]
    else:
        #print("Biggest value: ", v)
        v = list(set(v))
        return max(v)
        #return series.values[0]
#from src.utils.data import flatten_if_multiple
