package util;

import java.util.Random;

public class RandomUtil {
	public static final int RAND_SEED = 2016;
	public static Random rand = new Random (RAND_SEED);
	
	public static double nextDouble () {
		return rand.nextDouble();
	}
	public static double nextDouble (double a, double b) {
		if (a > b) {
			double tmp = a;
			a = b;
			b = tmp;
		}
		return rand.nextDouble() * (b - a) + a;
	}
}
