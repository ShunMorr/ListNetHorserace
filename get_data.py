from database import MysqlIO
import pandas as pd
import pickle


def get_data_from_race_id(race_id: int, mysql: MysqlIO):
    race_data = pd.DataFrame(mysql.select("t_race", "id", race_id))
    field = race_data[1]
    distance = race_data[4]
    horses_data = pd.DataFrame(mysql.select("t_record", "race_id", race_id))
    horse_data_result = []
    horse_order_result = []
    for _, horse_data in horses_data.iterrows():
        tmp_horse_name = horse_data[1]
        horse_order_result.append(horse_data[8])
        tmp_horse_data = []
        sql = f"Select * from t_horserecord where `horse_name`='{tmp_horse_name}' and `race_id`<{race_id} limit 5"
        tmp_data_list = mysql.execute(sql)
        if len(tmp_data_list) == 0:
            return False, horse_data_result, horse_order_result
        for tmp_data in tmp_data_list:
            tmp_horse_data.extend([tmp_data[3], round(tmp_data[5] / distance, 3)[0], round(tmp_data[6] * field, 3)[0]])
        tmp_horse_data = tmp_horse_data + [0] * (15 - len(tmp_horse_data))
        horse_data_result.append(tmp_horse_data)
    return True, horse_data_result, horse_order_result


if __name__ == "__main__":
    mysql = MysqlIO()
    race_list = mysql.select_all("t_race", "id")
    data_list = []
    order_list = []
    for i, race in enumerate(race_list):
        print(f"loading {race[0]} : {i}/{len(race_list)}")
        ret, data, order = get_data_from_race_id(race[0], mysql)
        if ret is False:
            continue
        data_list.append(data)
        order_list.append(order)
    with open("data_x.pkl", "wb") as f:
        pickle.dump(data_list, f)
    with open("data_y.pkl", "wb") as f:
        pickle.dump(order_list, f)
