from typing import List

import numpy as np

from a2t.legacy.relation_classification.mnli import (
    NLIRelationClassifierWithMappingHead,
)

TACRED_LABELS = [
    "no_relation",
    "org:alternate_names",
    "org:city_of_headquarters",
    "org:country_of_headquarters",
    "org:dissolved",
    "org:founded",
    "org:founded_by",
    "org:member_of",
    "org:members",
    "org:number_of_employees/members",
    "org:parents",
    "org:political/religious_affiliation",
    "org:shareholders",
    "org:stateorprovince_of_headquarters",
    "org:subsidiaries",
    "org:top_members/employees",
    "org:website",
    "per:age",
    "per:alternate_names",
    "per:cause_of_death",
    "per:charges",
    "per:children",
    "per:cities_of_residence",
    "per:city_of_birth",
    "per:city_of_death",
    "per:countries_of_residence",
    "per:country_of_birth",
    "per:country_of_death",
    "per:date_of_birth",
    "per:date_of_death",
    "per:employee_of",
    "per:origin",
    "per:other_family",
    "per:parents",
    "per:religion",
    "per:schools_attended",
    "per:siblings",
    "per:spouse",
    "per:stateorprovince_of_birth",
    "per:stateorprovince_of_death",
    "per:stateorprovinces_of_residence",
    "per:title",
]

TACRED_BASIC_LABELS = [
    "no_relation",
    "org:dissolved",
    "org:founded",
    "org:founded_by",
    "org:member_of",
    "org:members",
    "org:number_of_employees/members",
    "org:parents",
    "org:political/religious_affiliation",
    "org:shareholders",
    "org:subsidiaries",
    "org:top_members/employees",
    "org:website",
    "per:age",
    "per:cause_of_death",
    "per:charges",
    "per:children",
    "per:date_of_birth",
    "per:date_of_death",
    "per:employee_of",
    "per:other_family",
    "per:parents",
    "per:religion",
    "per:schools_attended",
    "per:siblings",
    "per:spouse",
    "per:title",
    "per:place_of_birth",
    "per:place_of_death",
    "per:place_of_residence",
    "org:place_of_headquarters",
    "both:alternate_names",
]

TACRED_BASIC_LABELS_MAPPING = {
    # Place of birth
    "per:city_of_birth": "per:place_of_birth",
    "per:country_of_birth": "per:place_of_birth",
    "per:stateorprovince_of_birth": "per:place_of_birth",
    # Origin can be see as a country/place of birth
    "per:origin": "per:place_of_birth",
    # Place of death
    "per:city_of_death": "per:place_of_death",
    "per:country_of_death": "per:place_of_death",
    "per:stateorprovince_of_death": "per:place_of_death",
    # Place of residence
    "per:cities_of_residence": "per:place_of_residence",
    "per:countries_of_residence": "per:place_of_residence",
    "per:stateorprovinces_of_residence": "per:place_of_residence",
    # Place of headquarters
    "org:city_of_headquarters": "org:place_of_headquarters",
    "org:country_of_headquarters": "org:place_of_headquarters",
    "org:stateorprovince_of_headquarters": "org:place_of_headquarters",
    # Alternative names
    "per:alternate_names": "both:alternate_names",
    "org:alternate_names": "both:alternate_names",
}

TACRED_LABEL_TEMPLATES = {
    "no_relation": ["{subj} and {obj} are not related"],
    "per:alternate_names": ["{subj} is also known as {obj}"],
    "per:date_of_birth": [
        "{subj}\u00e2\u0080\u0099s birthday is on {obj}",
        "{subj} was born in {obj}",
    ],
    "per:age": ["{subj} is {obj} years old"],
    "per:country_of_birth": ["{subj} was born in {obj}"],
    "per:stateorprovince_of_birth": ["{subj} was born in {obj}"],
    "per:city_of_birth": ["{subj} was born in {obj}"],
    "per:origin": ["{obj} is the nationality of {subj}"],
    "per:date_of_death": ["{subj} died in {obj}"],
    "per:country_of_death": ["{subj} died in {obj}"],
    "per:stateorprovince_of_death": ["{subj} died in {obj}"],
    "per:city_of_death": ["{subj} died in {obj}"],
    "per:cause_of_death": ["{obj} is the cause of {subj}’s death"],
    "per:countries_of_residence": [
        "{subj} lives in {obj}",
        "{subj} has a legal order to stay in {obj}",
    ],
    "per:stateorprovinces_of_residence": [
        "{subj} lives in {obj}",
        "{subj} has a legal order to stay in {obj}",
    ],
    "per:cities_of_residence": [
        "{subj} lives in {obj}",
        "{subj} has a legal order to stay in {obj}",
    ],
    "per:schools_attended": [
        "{subj} studied in {obj}",
        "{subj} graduated from {obj}",
    ],
    "per:title": ["{subj} is a {obj}"],
    "per:employee_of": [
        "{subj} is member of {obj}",
        "{subj} is an employee of {obj}",
    ],
    "per:religion": [
        "{subj} belongs to {obj} religion",
        "{obj} is the religion of {subj}",
        "{subj} believe in {obj}",
    ],
    "per:spouse": [
        "{subj} is the spouse of {obj}",
        "{subj} is the wife of {obj}",
        "{subj} is the husband of {obj}",
    ],
    "per:parents": [
        "{obj} is the parent of {subj}",
        "{obj} is the mother of {subj}",
        "{obj} is the father of {subj}",
        "{subj} is the son of {obj}",
        "{subj} is the daughter of {obj}",
    ],
    "per:children": [
        "{subj} is the parent of {obj}",
        "{subj} is the mother of {obj}",
        "{subj} is the father of {obj}",
        "{obj} is the son of {subj}",
        "{obj} is the daughter of {subj}",
    ],
    "per:siblings": [
        "{subj} and {obj} are siblings",
        "{subj} is brother of {obj}",
        "{subj} is sister of {obj}",
    ],
    "per:other_family": [
        "{subj} and {obj} are family",
        "{subj} is a brother in law of {obj}",
        "{subj} is a sister in law of {obj}",
        "{subj} is the cousin of {obj}",
        "{subj} is the uncle of {obj}",
        "{subj} is the aunt of {obj}",
        "{subj} is the grandparent of {obj}",
        "{subj} is the grandmother of {obj}",
        "{subj} is the grandson of {obj}",
        "{subj} is the granddaughter of {obj}",
    ],
    "per:charges": [
        "{subj} was convicted of {obj}",
        "{obj} are the charges of {subj}",
    ],
    "org:alternate_names": ["{subj} is also known as {obj}"],
    "org:political/religious_affiliation": [
        "{subj} has political affiliation with {obj}",
        "{subj} has religious affiliation with {obj}",
    ],
    "org:top_members/employees": [
        "{obj} is a high level member of {subj}",
        "{obj} is chairman of {subj}",
        "{obj} is president of {subj}",
        "{obj} is director of {subj}",
    ],
    "org:number_of_employees/members": [
        "{subj} employs nearly {obj} people",
        "{subj} has about {obj} employees",
    ],
    "org:members": ["{obj} is member of {subj}", "{obj} joined {subj}"],
    "org:member_of": ["{subj} is member of {obj}", "{subj} joined {obj}"],
    "org:subsidiaries": [
        "{obj} is a subsidiary of {subj}",
        "{obj} is a branch of {subj}",
    ],
    "org:parents": [
        "{subj} is a subsidiary of {obj}",
        "{subj} is a branch of {obj}",
    ],
    "org:founded_by": ["{subj} was founded by {obj}", "{obj} founded {subj}"],
    "org:founded": [
        "{subj} was founded in {obj}",
        "{subj} was formed in {obj}",
    ],
    "org:dissolved": [
        "{subj} existed until {obj}",
        "{subj} disbanded in {obj}",
        "{subj} dissolved in {obj}",
    ],
    "org:country_of_headquarters": [
        "{subj} has its headquarters in {obj}",
        "{subj} is located in {obj}",
    ],
    "org:stateorprovince_of_headquarters": [
        "{subj} has its headquarters in {obj}",
        "{subj} is located in {obj}",
    ],
    "org:city_of_headquarters": [
        "{subj} has its headquarters in {obj}",
        "{subj} is located in {obj}",
    ],
    "org:shareholders": ["{obj} holds shares in {subj}"],
    "org:website": [
        "{obj} is the URL of {subj}",
        "{obj} is the website of {subj}",
    ],
}

TACRED_VALID_CONDITIONS = {
    "per:alternate_names": ["PERSON:PERSON", "PERSON:MISC"],
    "per:date_of_birth": ["PERSON:DATE"],
    "per:age": ["PERSON:NUMBER", "PERSON:DURATION"],
    "per:country_of_birth": ["PERSON:COUNTRY"],
    "per:stateorprovince_of_birth": ["PERSON:STATE_OR_PROVINCE"],
    "per:city_of_birth": ["PERSON:CITY"],
    "per:origin": ["PERSON:NATIONALITY", "PERSON:COUNTRY", "PERSON:LOCATION"],
    "per:date_of_death": ["PERSON:DATE"],
    "per:country_of_death": ["PERSON:COUNTRY"],
    "per:stateorprovince_of_death": ["PERSON:STATE_OR_PROVICE"],
    "per:city_of_death": ["PERSON:CITY", "PERSON:LOCATION"],
    "per:cause_of_death": ["PERSON:CAUSE_OF_DEATH"],
    "per:countries_of_residence": ["PERSON:COUNTRY", "PERSON:NATIONALITY"],
    "per:stateorprovinces_of_residence": ["PERSON:STATE_OR_PROVINCE"],
    "per:cities_of_residence": ["PERSON:CITY", "PERSON:LOCATION"],
    "per:schools_attended": ["PERSON:ORGANIZATION"],
    "per:title": ["PERSON:TITLE"],
    "per:employee_of": ["PERSON:ORGANIZATION"],
    "per:religion": ["PERSON:RELIGION"],
    "per:spouse": ["PERSON:PERSON"],
    "per:parents": ["PERSON:PERSON"],
    "per:children": ["PERSON:PERSON"],
    "per:siblings": ["PERSON:PERSON"],
    "per:other_family": ["PERSON:PERSON"],
    "per:charges": ["PERSON:CRIMINAL_CHARGE"],
    "org:alternate_names": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:MISC"],
    "org:political/religious_affiliation": [
        "ORGANIZATION:RELIGION",
        "ORGANIZATION:IDEOLOGY",
    ],
    "org:top_members/employees": ["ORGANIZATION:PERSON"],
    "org:number_of_employees/members": ["ORGANIZATION:NUMBER"],
    "org:members": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:COUNTRY"],
    "org:member_of": [
        "ORGANIZATION:ORGANIZATION",
        "ORGANIZATION:COUNTRY",
        "ORGANIZATION:LOCATION",
        "ORGANIZATION:STATE_OR_PROVINCE",
    ],
    "org:subsidiaries": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:LOCATION"],
    "org:parents": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:COUNTRY"],
    "org:founded_by": ["ORGANIZATION:PERSON"],
    "org:founded": ["ORGANIZATION:DATE"],
    "org:dissolved": ["ORGANIZATION:DATE"],
    "org:country_of_headquarters": ["ORGANIZATION:COUNTRY"],
    "org:stateorprovince_of_headquarters": ["ORGANIZATION:STATE_OR_PROVINCE"],
    "org:city_of_headquarters": ["ORGANIZATION:CITY", "ORGANIZATION:LOCATION"],
    "org:shareholders": ["ORGANIZATION:PERSON", "ORGANIZATION:ORGANIZATION"],
    "org:website": ["ORGANIZATION:URL"],
}


class TACREDClassifier(NLIRelationClassifierWithMappingHead):
    def __init__(self, **kwargs):
        pretrained_model = kwargs.pop("pretrained_model", "microsoft/deberta-v2-xlarge-mnli")
        super(TACREDClassifier, self).__init__(
            pretrained_model=pretrained_model,
            labels=TACRED_LABELS,
            template_mapping=TACRED_LABEL_TEMPLATES,
            valid_conditions=TACRED_VALID_CONDITIONS,
            entailment_position=2,
            **kwargs
        )
