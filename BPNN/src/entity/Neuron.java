package entity;

/**
 * 神经元
 * @author huangyundu
 *
 */
public class Neuron {
	
	/**
	 * 神经元值
	 */
	public double value;
	/**
	 * 神经元输出值
	 */
	public double o;
	
	public Neuron () {
		init ();
	}
	public Neuron (double v) {
		init (v);
	}
	public Neuron (double v, double o) {
		this.value = v;
		this.o = o;
	}
	
	public void init () {
		this.value = 0;
		this.o = 0;
	}
	public void init (double v) {
		this.value = v;
		this.o = 0;
	}
	public void init (double v, double o) {
		this.value = v;
		this.o = o;
	}
	
	/**
	 * sigmod激活函数
	 */
	public void sigmod () {
		this.o = 1.0 / ( 1.0 + Math.exp(-1.0 * this.value));
	}
	
	public String toString () {
		return "(" + value + " " + o + ")";
	}
}
