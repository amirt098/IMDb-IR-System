
from typing import List
import wandb


class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        for act, pred in zip(actual, predicted):
            if len(pred) == 0:
                precision += 0.0
                continue

            relevant_retrieved = set(act).intersection(set(pred))
            precision += len(relevant_retrieved) / len(pred)

        precision = precision / len(actual)

        return precision
        

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        for act, pred in zip(actual, predicted):
            if len(act) == 0:
                recall += 0.0
                continue

            relevant_retrieved = set(act).intersection(set(pred))
            recall += len(relevant_retrieved) / len(act)

        recall = recall / len(actual)
        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)

        if precision + recall == 0:
            return f1
        f1 = 2 * (precision * recall) / (precision + recall)

        return f1
    
    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0

        for act, pred in zip(actual, predicted):
            if not act:
                continue

            score = 0.0
            num_relevant = 0

            for i, p in enumerate(pred):
                if p in act:
                    num_relevant += 1
                    score += num_relevant / (i + 1)

            AP += score / len(act)
        AP =  AP / len(actual)

        return AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        def average_precision(act, pred):
            if not act:
                return 0.0
            score = 0.0
            num_relevant = 0
            for i, p in enumerate(pred):
                if p in act:
                    num_relevant += 1
                    score += num_relevant / (i + 1)
            return score / len(act)

        total_ap = 0.0
        for act, pred in zip(actual, predicted):
            total_ap += average_precision(act, pred)

        MAP = total_ap / len(actual)

        return MAP

    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0


        for i, p in enumerate(predicted):
            if p in actual:
                DCG += 1 / (i + 1)

        return DCG

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        for act, pred in zip(actual, predicted):
            dcg = self.cacluate_DCG(act, pred)
            ideal_dcg = self.cacluate_DCG(act, act)
            if ideal_dcg == 0:
                NDCG += 0.0
            else:
                NDCG += dcg / ideal_dcg
        NDCG = NDCG / len(actual)

        return NDCG
    
    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = 0.0

        # TODO: Calculate MRR here
        for i, p in enumerate(predicted):
            if p in actual:
                RR = 1 / (i + 1)

        return RR
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0

        for act, pred in zip(actual, predicted):
            MRR += self.cacluate_RR(act, pred)
        MRR = MRR / len(actual)

        return MRR
    

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(f"average precision - AP: {ap:.4f}")
        print(f"MAP: {map:.4f}")
        print(f"DCG - Discounted Cumulative Gain: {dcg:.4f}")
        print(f"Normalized Discounted Cumulative Gain - NDCG: {ndcg:.4f}")
        print(f"Reciprocal Rank - RR: {rr:.4f}")
        print(f"Mean Reciprocal Rank - MRR: {mrr:.4f}")
      

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        

        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Average Precision (AP)": ap,
            "Mean Average Precision (MAP)": map,
            "Discounted Cumulative Gain (DCG)": dcg,
            "Normalized Discounted Cumulative Gain (NDCG)": ndcg,
            "Reciprocal Rank (RR)": rr,
            "Mean Reciprocal Rank (MRR)": mrr
        })

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)


#
# if __name__ == "__main__":
#     actual = [
#         ["doc1", "doc2", "doc3"],
#         ["doc1", "doc4", "doc5"],
#         ["doc2", "doc3"]
#     ]
#
#     predicted = [
#         ["doc1", "doc4", "doc3"],
#         ["doc1", "doc2", "doc3"],
#         ["doc3", "doc2"]
#     ]
#     # wandb.init(project="evaluation-metrics")
#
#     evaluator = Evaluation(name="TestEvaluation")
#     evaluator.calculate_evaluation(actual, predicted)
#
