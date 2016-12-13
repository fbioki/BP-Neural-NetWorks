package entity;

import java.util.ArrayList;
import java.util.List;

import util.RandomUtil;

public class NeuronNet {
	/**
	 * 神经网络
	 */
	public List<List<Neuron>> neurons;
	/**
	 * 网络权重, weight[i][j][k] = 第i层和第i+1层之间, j神经元和k神经元的权重, i = 0 ... layer-1
	 */
	public double [][][] weight;
	/**
	 * 下一层网络残差, deta[i][j] = 第i层, 第j神经元的残差, i = 1 ... layer-1
	 */
	public double [][] deta;
	
	/**
	 * 网络层数（包括输入与输出层）
	 */
	public int layer;
	/**
	 * 学习率
	 */
	public static final double RATE = 0.1;
	/**
	 * 误差停止阈值
	 */
	public static final double STOP = 0.0001;
	/**
	 * 迭代次数阈值
	 */
	public static final int NUMBER_ROUND = 5000000;
	
	public NeuronNet () {
		
	}
	public NeuronNet (int [] lens) {
		init (lens);
	}
	
	public void init (int [] lens) {
		layer = lens.length;
		neurons = new ArrayList<List<Neuron>> ();
		
		for (int i = 0; i < layer; ++ i) {
			List<Neuron> list = new ArrayList<Neuron> ();
			for (int j = 0; j < lens[i]; ++ j) {
				list.add(new Neuron ());
			}
			neurons.add(list);
		}
		
		weight = new double [layer-1][][];
		for (int i = 0; i < layer-1; ++ i) {
			weight[i] = new double [lens[i]][];
			for (int j = 0; j < lens[i]; ++ j) {
				weight[i][j] = new double [lens[i+1]];
				for (int k = 0; k < lens[i+1]; ++ k) {
					weight[i][j][k] = RandomUtil.nextDouble(0, 0.1);
				}
			}
		}
		
		deta = new double [layer][];
		for (int i = 0; i < layer; ++ i) {
			deta[i] = new double [lens[i]];
			for (int j = 0; j < lens[i]; ++ j) deta[i][j] = 0;
		}
	}
	
	/**
	 * 前向传播
	 * @param features
	 * @return
	 */
	public boolean forward (double [] features) {
		// c = layer index
		for (int c = 0; c < layer; ++ c) {
			if (c == 0) {
				// 初始化输入层
				
				List<Neuron> inputLayer = neurons.get(c);
				if (inputLayer.size() != features.length) {
					System.err.println("[error] Feature length != input layer neuron number");
					return false;
				}
				for (int i = 0; i < inputLayer.size(); ++ i) {
					Neuron neuron = inputLayer.get(i);
					neuron.init(features[i], features[i]);
				}
			} else {
				// 前向传播：从c-1层传播到c层
				List<Neuron> vList = neurons.get(c);
				List<Neuron> uList = neurons.get(c-1);
				for (int i = 0; i < vList.size(); ++ i) {
					Neuron v = vList.get(i);
					v.value = 0;
					for (int j = 0; j < uList.size(); ++ j) {
						Neuron u = uList.get(j);
						v.value += u.o * weight[c-1][j][i];
					}
					v.sigmod();
				}
			}
		}
		return true;
	}
	
	/**
	 * 求误差函数
	 * @param labels 期望输出层向量
	 * @return
	 */
	public double getError (double [] labels) {
		if (labels.length != neurons.get(layer-1).size()) {
			System.err.println("[error] label length != output layer neuron number");
			return -1;
		}
		double e = 0;
		for (int i = 0; i < labels.length; ++ i) {
			double o = neurons.get(layer-1).get(i).o;
			e += (labels[i] - o) * (labels[i] - o);
		}
		return e / 2;
	}
	/**
	 * 获取输出层向量
	 * @return
	 */
	public double[] getOutput () {
		double [] output = new double [neurons.get(layer-1).size()];
		for (int i = output.length-1; i >= 0 ; -- i)
			output [i] = neurons.get(layer-1).get(i).o;
		return output;
	}
	
	/**
	 * 反向传播
	 * @param labels
	 * @return
	 */
	public boolean backward (double [] labels) {
		if (labels.length != neurons.get(layer-1).size()) {
			System.err.println("[error] label length != output layer neuron number");
			return false;
		}
		// 初始化output层(layer-1)残差
		for (int j = neurons.get(layer-1).size()-1; j >= 0 ; -- j) {
			double o = neurons.get(layer-1).get(j).o;
			// 求导公式
			deta[layer-1][j] = -1 * (labels[j] - o) * o * (1 - o);
			// 更新倒数第二层和最后一层之间的权重
			for (int i = neurons.get(layer-2).size()-1; i >= 0; -- i) {
				weight[layer-2][i][j] -= RATE * deta[layer-1][j] * neurons.get(layer-2).get(i).o;
			}
		}
		//A层（layer=L）和B层（layer=L+1）间权重调整，用到了C层（layer=L+2）
		for (int l = layer-3; l >= 0; -- l) {
			
			// 遍历B层
			for (int j = neurons.get(l+1).size()-1; j >= 0; -- j) {
				// B层J神经元残差
				deta[l+1][j] = 0;
				// 遍历C层, 求残差和
				for (int k = neurons.get(l+2).size()-1; k >= 0; -- k) {
					// C层残差通过权重weight传递过来
					deta[l+1][j] += deta[l+2][k] * weight[l+1][j][k];
				}
				double o = neurons.get(l+1).get(j).o;
				deta[l+1][j] *= o * (1 - o);
				
				// 遍历A层
				for (int i = neurons.get(l).size()-1; i >= 0; -- i) {
					// A层i神经元和B层j神经元权重
					weight[l][i][j] -= RATE * deta[l+1][j] * neurons.get(l).get(i).o;
				}
			}
		}
		return true;
	}
	
	public void train (double [][] features, double [][] labels) {
		SGD (features, labels);
	}
	
	public void SGD (double [][] features, double [][] labels) {
		int num = 0;
		double error = 1;
		while ((num ++) <= NUMBER_ROUND && error > STOP) {
			for (int i = 0; i < features.length; ++ i) {
				boolean flag = forward (features[i]);
				if (!flag) {
					return;
				}
				
				error = this.getError(labels[i]);
				if (error == -1) {
					return;
				}
				
				if (error <= STOP)
					break;
				
				flag = backward (labels[i]);
				if (!flag) {
					return;
				}
			}
			System.out.println("[Info] Times = " + num + ", error = " + error);
		}
	}
	
	public double [] predict (double [] feature) {
		forward (feature);
		return this.getOutput();
	}
}