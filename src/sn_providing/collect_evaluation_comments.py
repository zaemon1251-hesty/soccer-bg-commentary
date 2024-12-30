import os

from tap import Tap
import pandas as pd

from sn_providing.entity import SpottingDataList, ReferenceDoc


class Arguments(Tap):
    input_a_file: str
    input_b_file: str
    input_c_file: str
    input_a_star_file: str
    input_b_star_file: str
    input_c_star_file: str
    output_file: str
    reference_documents_yaml: str 


def main(args: Arguments):
    reference_doc_list = ReferenceDoc.get_list_from_yaml(args.reference_documents_yaml)
    
    input_a_data_list = SpottingDataList.from_jsonline(args.input_a_file)
    input_b_data_list = SpottingDataList.from_jsonline(args.input_b_file)
    input_c_data_list = SpottingDataList.from_jsonline(args.input_c_file)
    input_a_star_data_list = SpottingDataList.from_jsonline(args.input_a_star_file)
    input_b_star_data_list = SpottingDataList.from_jsonline(args.input_b_star_file)
    input_c_star_data_list = SpottingDataList.from_jsonline(args.input_c_star_file)
    
    all_data_dict = {
        "a": input_a_data_list,
        "b": input_b_data_list,
        "c": input_c_data_list,
        "a_star": input_a_star_data_list,
        "b_star": input_b_star_data_list,
        "c_star": input_c_star_data_list
    }
    
    result_list = []
    
    for approach, data_list in all_data_dict.items():
        for spotting_data in data_list.spottings:
            reference_doc = ReferenceDoc.get_reference_document_entity(
                game=spotting_data.game,
                half=spotting_data.half,
                time=spotting_data.game_time,
                reference_documents=reference_doc_list,
            )
            if reference_doc is not None:
                sample_id = reference_doc.id
                result_list.append({"sample_id": sample_id,
                                    "approach": approach,
                                    "generated_text": spotting_data.generated_text,
                                    "game": spotting_data.game,
                                    "half": spotting_data.half,
                                    "time": spotting_data.game_time})

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    result_df = pd.DataFrame(result_list).sort_values(by=["sample_id", "approach"])
    result_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
