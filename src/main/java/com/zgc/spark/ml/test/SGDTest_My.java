package com.zgc.spark.ml.test;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * 随机梯度下降模拟
 *
 * @author guocheng.zhao
 * @date 2017/6/8 11:26
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */
public class SGDTest_My {
    public static double theta = 0f;
    public static double af = 0.1f;

    public static void main(String[] args) {
        Map<Integer, Integer> data = getData();
        Iterator<Integer> it = data.keySet().iterator();
        while (it.hasNext()){
            Integer k = it.next();
            Integer v = data.get(k);
            sgd(k,v);
        }
        /*data.forEach((k, v) -> {
            sgd(k,v);
        });*/
        System.out.println("theta:"+theta);
    }

    /**
     * 梯度下降
     *
     * @param x
     * @param y
     */
    public static void sgd(double x, double y) {
        theta = theta - af * ((x * theta) - y);
    }

    public static Map<Integer, Integer> getData() {
        Map<Integer, Integer> data = new HashMap<>();
        for (int i = 1; i <= 50; i++) {
            data.put(i, (i * 11)+1);
        }
        return data;
    }
}
