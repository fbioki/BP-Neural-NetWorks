package main;

import entity.NeuronNet;

public class Main {
	public static void main(String[] args) {
		// 三层神经网络，每层神经元个数分别是3，5，8
		NeuronNet bpnn = new NeuronNet (new int [] {3, 5, 8});
		
		// 数据说明，求二进制X[i]的十进制表示Y[i]
		double[][] X = {
				{0,0,0},
				{0,0,1},
				{0,1,0},
				{0,1,1},
				{1,0,0},
				{1,0,1},
				{1,1,0},
				{1,1,1}
		};
		double [][] Y = {
				{1, 0, 0, 0, 0, 0, 0, 0},
				{0, 1, 0, 0, 0, 0, 0, 0},
				{0, 0, 1, 0, 0, 0, 0, 0},
				{0, 0, 0, 1, 0, 0, 0, 0},
				{0, 0, 0, 0, 1, 0, 0, 0},
				{0, 0, 0, 0, 0, 1, 0, 0},
				{0, 0, 0, 0, 0, 0, 1, 0},
				{0, 0, 0, 0, 0, 0, 0, 1}
		};
		
		bpnn.train(X, Y);
		
		for (int i = 0; i < 8; ++ i) {
			double [] output = bpnn.predict(X[i]);
			double max = -1;
			int pos = -1;
			// 求最接近的神经元
			for (int j = 0; j < output.length; ++ j) {
				if (max < output[j]) {
					max = output[j];
					pos = j;
				}
			}
			System.out.print (X[i][0]);
			for (int j = 1; j < X[i].length; ++ j) {
				System.out.print (", " + X[i][j]);
			}
			System.out.println(" = " + pos);
		}
	}
}
