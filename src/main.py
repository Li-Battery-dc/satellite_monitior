from fault_detection.random_forest import RandomForestDetector
from fault_detection.MLP import MLPDetector

from utils.data_loader import Dataloader
from utils.evaluation import Evaluator

def main():
    data_loader = Dataloader(data_root="../data/train", data_name='激光载荷')
    data = data_loader.get_data('激光载荷')
    
    X_train = data['X_train']
    y_train = data['y_train']
    y_train[y_train != 0] = 1  # 转换为二分类问题


    X_test = data['X_test']
    y_test = data['y_test']
    y_test[y_test != 0] = 1  # 转换为二分类问题
    
    # 初始化随机森林故障检测器
    # detector = RandomForestDetector()

    # 初始化MLP故障检测器
    detector = MLPDetector(input_dim=X_train.shape[1], hidden_layers=[64, 32])
    
    # 训练模型  
    detector.fit(X_train, y_train)
    
    # 评估模型
    y_pred = detector.predict(X_test)
    y_prob = detector.predict_prob(X_test)
    evaluator = Evaluator(y_test, y_pred, y_prob)
    evaluator.print_results()

if __name__ == "__main__":
    main()