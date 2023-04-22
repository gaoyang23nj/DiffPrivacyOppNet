package routing;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;

import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.apache.poi.xssf.usermodel.XSSFRow;

import core.Connection;
import core.DTNHost;
import core.Message;
import core.Settings;
import core.SimClock;
import util.Tuple;


//RTPMSpdUpDjkRouter TTPM
/**
 * Implementation of TTPM as described in
 * <I>This paper</I> by
 * Yang Gao et al.
 */
public class TTPMSpdUpDjkRouter extends ActiveRouter {
	/**
	 * number of nodes
	 */
	int number_nodes;
	/**
	 * number of segs of {p}
	 */
	int number_segs;
	
	/**
	 * current index of day, 1,2,3... ...
	 */
	int current_day_index;
	
//	// on the current day
//	// dimension: info holder's node_id (the view), from_node_id, to_node_id 
//	double[][][] BigP_holiday;
//	double[][][] BigP_workday;
//	
//	// dimension: info holder's node_id (the view), from_node_id, to_node_id, seg_id
//	double[][][][] smallp_holiday;
//	double[][][][] smallp_workday;
	
	// on the current day
	// dimension: info holder's node_id (the view), from_node_id, to_node_id 
	double[][] BigP_holiday;
	double[][] BigP_workday;
	
	// dimension: info holder's node_id (the view), from_node_id, to_node_id, seg_id
	double[][][] smallp_holiday;
	double[][][] smallp_workday;
	
	double[][][] all_res_cal;
	boolean[] listHolidayinYear;
	
	int max_ttl_day;
	int max_sim_day;
	
	/**
	 * constructor 
	 * @param s
	 */
	public TTPMSpdUpDjkRouter(Settings s) {
		super(s);
		// TODO Auto-generated constructor stub
		// read number of stations; read time
		this.number_nodes = 99;
		// considering P and {p} should be updated every day
		this.max_sim_day = 14;
		//
		this.number_segs = 24;
		//max ttl day
		this.max_ttl_day = 5;
		this.current_day_index = 0;
		
		this.BigP_holiday = new double[this.number_nodes][this.number_nodes];
		this.BigP_workday = new double[this.number_nodes][this.number_nodes];
		this.smallp_holiday = new double[this.number_nodes][this.number_nodes][this.number_segs];
		this.smallp_workday = new double[this.number_nodes][this.number_nodes][this.number_segs];
		
		this.all_res_cal = new double[this.number_nodes][this.number_nodes][(this.max_ttl_day + 1)*this.number_segs];
		
		this.listHolidayinYear = new boolean[365];
		this.readDayInfo();
	}

	/**
	 * Copyconstructor.
	 * @param r The router prototype where setting values are copied from
	 */
	protected TTPMSpdUpDjkRouter(TTPMSpdUpDjkRouter r) {
		super(r);
		this.number_nodes = r.number_nodes;
		// considering P and {p} should be updated every day
		this.max_sim_day = r.max_sim_day;
		//
		this.number_segs = r.number_segs;
		//max ttl day
		this.max_ttl_day = r.max_ttl_day;
		this.current_day_index = r.current_day_index;
				
		this.BigP_holiday = r.BigP_holiday;
		this.BigP_workday = r.BigP_workday;
		this.smallp_holiday = r.smallp_holiday;
		this.smallp_workday = r.smallp_workday;
		this.all_res_cal = r.all_res_cal;
		this.listHolidayinYear = r.listHolidayinYear;
		
	}
	
	@Override
	public MessageRouter replicate() {
		TTPMSpdUpDjkRouter r = new TTPMSpdUpDjkRouter(this);
		return r;
	}
	
	private void readDayInfo() {
		String folderpath = ".//NanjingData//";
		String filename = folderpath + "Pukou_Weather.xlsx";
		try {
	        XSSFWorkbook xssfWorkbook = new XSSFWorkbook(new FileInputStream(filename));
	        //读取第一个工作表
	        XSSFSheet sheet = xssfWorkbook.getSheetAt(0);
	        //总行数
	        int maxRow = sheet.getLastRowNum();
	        System.out.println(maxRow);
	        //skip the first line; it's the table header
			for (int row = 1; row <= maxRow; row++) {
				
				XSSFRow line = sheet.getRow(row);
				int maxColumn = line.getLastCellNum();
				int isHoliday = -1;
				if (maxColumn==7) {
					isHoliday = (int) Double.parseDouble(line.getCell(maxColumn-1)+"");
				}
				boolean isWeekend = ((line.getCell(0)+"").contains("周六") || (line.getCell(0)+"").contains("周日"));
				if (isHoliday == 1) {
					this.listHolidayinYear[row-1] = true;
				}else if (isHoliday == 0) {
					this.listHolidayinYear[row-1] = false;
				}else if (isWeekend == true) {
					this.listHolidayinYear[row-1] = true;
				}else{
					this.listHolidayinYear[row-1] = false;
				}
//				System.out.println(this.listHolidayinYear[row-1]);
//				System.out.println();	
			}
			System.out.println("xlsx文件查询完成");
	        xssfWorkbook.close();
	        
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * update P and {p} every day
	 * @param k  No. of days
	 */
	public void getInfo(int k) {
//		System.out.println(String.format("read file at k:%d", k));
		// from 1 to 31
		//int index_day = t.getDayOfMonth()-1;
		String folderpath = "..//..//01-code//Main//PandpforOne//";
		int view_node_id = this.getHost().getAddress();
		String tmp = String.format("node_%d_day_%d.csv", view_node_id, k);
		
		// load P holiday
		String filename = folderpath + "BigP_holiday_" + tmp;
	    try {
	        FileReader filereader = new FileReader(filename);
	        BufferedReader bufferedReader = new BufferedReader(filereader);
	        String strLine = null;
	        int lineCount = 0;
	        int from_node = -1;
	        while(null != (strLine = bufferedReader.readLine())){
	        	if(strLine.contains("from_node")){
	        		String[] tmp_strLine = strLine.split(":");
	        		from_node = Integer.parseInt(tmp_strLine[tmp_strLine.length-1]);
	        	}else{
	        		String [] tmp_str = strLine.split(",");
	        		assert(tmp_str.length == this.number_nodes);
	        		for (int i=0; i<this.number_nodes; i++) {
	        			 this.BigP_holiday[from_node][i] = Double.parseDouble(tmp_str[i]);
	        		}
		        	// System.out.println(strLine);		        		
	        	}
	            lineCount++;
	        }
	        bufferedReader.close();
	        filereader.close();
	    }catch(Exception e){
	        e.printStackTrace();
	    }

		// load P workday
		filename = folderpath + "BigP_workday_" + tmp;
	    try {
	        FileReader filereader = new FileReader(filename);
	        BufferedReader bufferedReader = new BufferedReader(filereader);
	        String strLine = null;
	        int lineCount = 0;
	        int from_node = -1;
	        while(null != (strLine = bufferedReader.readLine())){
	        	if(strLine.contains("from_node")){
	        		String[] tmp_strLine = strLine.split(":");
	        		from_node = Integer.parseInt(tmp_strLine[tmp_strLine.length-1]);
	        	}else{
	        		String [] tmp_str = strLine.split(",");
	        		assert(tmp_str.length == this.number_nodes);
	        		for (int i=0; i<this.number_nodes; i++) {
	        			 this.BigP_workday[from_node][i] = Double.parseDouble(tmp_str[i]);
	        		}
		        	// System.out.println(strLine);		        		
	        	}
	            lineCount++;
	        }
	        bufferedReader.close();
	        filereader.close();
	    }catch(Exception e){
	        e.printStackTrace();
	    }
		
		// load {p} holiday
		filename = folderpath + "smallp_holiday_" + tmp;		
	    try {
	        FileReader filereader = new FileReader(filename);
	        BufferedReader bufferedReader = new BufferedReader(filereader);
	        String strLine = null;
	        int lineCount = 0;
	        int from_node = -1;
	        int to_node = -1;
	        while(null != (strLine = bufferedReader.readLine())){
	        	if(strLine.contains("from_node")){
	        		String[] tmp_strLine = strLine.split(":");
	        		from_node = Integer.parseInt(tmp_strLine[tmp_strLine.length-1]);
	        	}else if(strLine.contains("to_node")){
	        		String[] tmp_strLine = strLine.split(":");
	        		to_node = Integer.parseInt(tmp_strLine[tmp_strLine.length-1]);
	        	}else{
	        		String [] tmp_str = strLine.split(",");
	        		assert(tmp_str.length == this.number_segs);
	        		for (int i=0; i<this.number_segs; i++) {
	        			 this.smallp_holiday[from_node][to_node][i] = Double.parseDouble(tmp_str[i]);
	        		}
		        	// System.out.println(strLine);		        		
	        	}
	            lineCount++;
	        }
	        bufferedReader.close();
	        filereader.close();
	    }catch(Exception e){
	        e.printStackTrace();
	    }
		
		// load {p} workday
		filename = folderpath + "smallp_workday_" + tmp;	
	    try {
	        FileReader filereader = new FileReader(filename);
	        BufferedReader bufferedReader = new BufferedReader(filereader);
	        String strLine = null;
	        int lineCount = 0;
	        int from_node = -1;
	        int to_node = -1;
	        while(null != (strLine = bufferedReader.readLine())){
	        	if(strLine.contains("from_node")){
	        		String[] tmp_strLine = strLine.split(":");
	        		from_node = Integer.parseInt(tmp_strLine[tmp_strLine.length-1]);
	        	}else if(strLine.contains("to_node")){
	        		String[] tmp_strLine = strLine.split(":");
	        		to_node = Integer.parseInt(tmp_strLine[tmp_strLine.length-1]);
	        	}else{
	        		String [] tmp_str = strLine.split(",");
	        		assert(tmp_str.length == this.number_segs);
	        		for (int i=0; i<this.number_segs; i++) {
	        			 this.smallp_workday[from_node][to_node][i] = Double.parseDouble(tmp_str[i]);
	        		}
		        	// System.out.println(strLine);		        		
	        	}
	            lineCount++;
	        }
	        bufferedReader.close();
	        filereader.close();
	    }catch(Exception e){
	        e.printStackTrace();
	    }
		
	    if (this.getHost().getAddress()==0) {
			System.out.println(String.format("**** Load P and {p} on %d day at %d end!!", k, this.getHost().getAddress()));	    	
	    }
	}
	
	@Override
	public void update() {
		super.update();
		
		// if new day is comming, update Info
		// the updating process of 'this.current_day_index':
		// when 20170701 begins, =0, simclock>0, getInfo 1, dayindex 0, update to be 1;
		// when 20170702 begins, =1, simclock>1, getInfo 2, dayindex 1, update to be 2;
		// =2, ...
		//(1,209,600=3600*24*14)
		if (SimClock.getTime() >= this.current_day_index*60*60*24) {
			//first one when day_index=1, .... when day_index=14.
			if ((this.current_day_index>=0) && (this.current_day_index<=13)) {
				this.getInfo(this.current_day_index+1);
				this.MaintainResCal(this.current_day_index);
			}else {
				if (this.getHost().getAddress() == 3) {
					System.out.println(String.format("day_index: %d at %d", this.current_day_index, SimClock.getIntTime()));					
				}
			}
			this.current_day_index = this.current_day_index + 1;
		}	
		
		if (!canStartTransfer() ||isTransferring()) {
			return; // nothing to transfer or is currently transferring
		}

		// try messages that could be delivered to final recipient
		if (exchangeDeliverableMessages() != null) {
			return;
		}
		
		this.tryOtherMessages();
	}

	private Tuple<Message, Connection> tryOtherMessages() {
		List<Tuple<Message, Connection>> messages =
			new ArrayList<Tuple<Message, Connection>>();

		Collection<Message> msgCollection = getMessageCollection();

		/* for all connected hosts collect all messages that have a higher
		   probability of delivery by the other host */
		for (Connection con : getConnections()) {
			DTNHost other = con.getOtherNode(getHost());
			TTPMSpdUpDjkRouter othRouter = (TTPMSpdUpDjkRouter)other.getRouter();

			if (othRouter.isTransferring()) {
				continue; // skip hosts that are transferring
			}

			for (Message m : msgCollection) {
				if (othRouter.hasMessage(m.getId())) {
					continue; // skip messages that the other one has
				}
				//important delivery metric comparision
				if (othRouter.getDeliveryMetric(m) > getDeliveryMetric(m)) {
					// the other node has higher probability of delivery
					messages.add(new Tuple<Message, Connection>(m,con));
				}
			}
		}

		if (messages.size() == 0) {
			return null;
		}

		// sort the message-connection tuples
		Collections.sort(messages, new TupleComparator());
		return tryMessagesForConnected(messages);	// try to send messages
	}
	
	/**
	 * Comparator for Message-Connection-Tuples that orders the tuples by
	 * their delivery probability by the host on the other side of the
	 * connection (GRTRMax)
	 */
	private class TupleComparator implements Comparator
		<Tuple<Message, Connection>> {

		public int compare(Tuple<Message, Connection> tuple1,
				Tuple<Message, Connection> tuple2) {
			// delivery probability of tuple1's message with tuple1's connection
			double p1 = ((TTPMSpdUpDjkRouter)tuple1.getValue().
					getOtherNode(getHost()).getRouter()).getDeliveryMetric(
					tuple1.getKey());
			// -"- tuple2...
			double p2 = ((TTPMSpdUpDjkRouter)tuple2.getValue().
					getOtherNode(getHost()).getRouter()).getDeliveryMetric(
					tuple2.getKey());

			// bigger probability should come first
			if (p2-p1 == 0) {
				/* equal probabilities -> let queue mode decide */
				return compareByQueueMode(tuple1.getKey(), tuple2.getKey());
			}
			else if (p2-p1 < 0) {
				return -1;
			}
			else {
				return 1;
			}
		}
	}
	
	/**
	 * 	
	 * @param k the day in the year, from 0 to 13, two weeks.
	 */
	private void MaintainResCal(int k) {
		//P*(1-P)*(1-P)
		double[][][] tmp_rescal = new double[this.number_nodes][this.number_nodes][(this.max_ttl_day+1)*this.number_segs];
		double[][][] cond_P = new double[this.number_nodes][this.number_nodes][this.max_ttl_day+1];
		
		//init day index 2017-07-01
		Calendar cd = Calendar.getInstance();
		cd.set(2017, 6, 1);
		int day_index = cd.get(Calendar.DAY_OF_YEAR)-1;
		assert(day_index == 181);
		//this is k-th day in the simulation
		day_index = day_index + k;
		
		//for {p} in tmp_rescal
		for (int target_i=0; target_i < this.max_ttl_day+1; target_i ++) {
			//0,24,48
			int from_y_index = target_i * this.number_segs;
			//24,48
//			int to_y_index = (target_i+1) * this.number_segs;
			//0,... 23
			if (this.listHolidayinYear[day_index+target_i] == true) {
				for (int tmp_i = 0; tmp_i < this.number_segs; tmp_i++) {
					for (int from_node_id = 0; from_node_id < this.number_nodes; from_node_id++) {
						for (int to_node_id = 0; to_node_id < this.number_nodes; to_node_id++) {
							tmp_rescal[from_node_id][to_node_id][from_y_index + tmp_i] = 
									this.smallp_holiday[from_node_id][to_node_id][tmp_i];						
						}
					}
				}
			}else {
				for (int tmp_i = 0; tmp_i < this.number_segs; tmp_i++) {
					for (int from_node_id = 0; from_node_id < this.number_nodes; from_node_id++) {
						for (int to_node_id = 0; to_node_id < this.number_nodes; to_node_id++) {
							tmp_rescal[from_node_id][to_node_id][from_y_index + tmp_i] = 
									this.smallp_workday[from_node_id][to_node_id][tmp_i];						
						}
					}
				}
			}
		}
		
//		Date dt = new Date();
//		dt.get
		// for P in cond_P
		for (int from_node_id = 0; from_node_id < this.number_nodes; from_node_id++) {
			for (int to_node_id = 0; to_node_id < this.number_nodes; to_node_id++) {
				// occurs on target_i's day
				for (int target_i=0; target_i < this.max_ttl_day+1; target_i ++) {
					cond_P[from_node_id][to_node_id][target_i] = 1.0;
					for (int from_i=0; from_i < target_i; from_i++) {
						if(this.listHolidayinYear[day_index + from_i] == true) {
							cond_P[from_node_id][to_node_id][target_i] = cond_P[from_node_id][to_node_id][target_i] *
									(1-this.BigP_holiday[from_node_id][to_node_id]);
						}else {
							cond_P[from_node_id][to_node_id][target_i] = cond_P[from_node_id][to_node_id][target_i] *
									(1-this.BigP_workday[from_node_id][to_node_id]);
						}

					}
					if(this.listHolidayinYear[day_index + target_i] == true) {
						cond_P[from_node_id][to_node_id][target_i] = cond_P[from_node_id][to_node_id][target_i] * 
								this.BigP_holiday[from_node_id][to_node_id];
					}else {
						cond_P[from_node_id][to_node_id][target_i] = cond_P[from_node_id][to_node_id][target_i] * 
								this.BigP_workday[from_node_id][to_node_id];			
					}
				}
			}
		}
		
		// calculate res_cal, (1-P)*(1-P)*P *[,seg,,,]
		for (int from_node_id = 0; from_node_id < this.number_nodes; from_node_id++) {
			for (int to_node_id = 0; to_node_id < this.number_nodes; to_node_id++) {
				for (int target_day_i=0; target_day_i < this.max_ttl_day+1; target_day_i ++) {
					for (int target_seg_i=0; target_seg_i < this.number_segs; target_seg_i ++) {
						int index = target_day_i * this.number_segs + target_seg_i;
						this.all_res_cal[from_node_id][to_node_id][index] =
								cond_P[from_node_id][to_node_id][target_day_i] *
								tmp_rescal[from_node_id][to_node_id][index];
						
					}
				}
			}	
		}		
	}
	
	private double getDeliveryMetric(Message m) {
		int time_slice_length = (60*60*24)/this.number_segs;
		// current seconds in this day (); 2017-07-01, 2017-07-02, ...
		int tmp_time = SimClock.getIntTime();
		// when 2017-07-01, this.current_day_index==1
		int current_time = tmp_time - (this.current_day_index-1)*60*60*24;
		int current_index = current_time/time_slice_length;
		
		double delta_time =  m.getTtl()*60.;
		//1209600
		if (SimClock.getIntTime() + delta_time>=this.max_sim_day*24*60*60) {
			delta_time = this.max_sim_day*24*60*60 - SimClock.getIntTime();
		}
		
		int end_index = (int) Math.ceil(delta_time/time_slice_length);
		
//		System.out.println(String.format("calculate metric, at node=%d, SimClock=%d, day_k=%d, from %d to %d",
//				this.getHost().getAddress(), tmp_time, this.current_day_index, current_index, end_index));
		
		//1. calculate weight for each single hop;
		double[][] matrix = new double[this.number_nodes][this.number_nodes];
		for (int from_node_id = 0; from_node_id < this.number_nodes; from_node_id++) {
			for (int to_node_id = 0; to_node_id < this.number_nodes; to_node_id++) {
				matrix[from_node_id][to_node_id] = Double.MAX_VALUE;
			}		
		}
		for (int from_node_id = 0; from_node_id < this.number_nodes; from_node_id++) {
			for (int to_node_id = 0; to_node_id < this.number_nodes; to_node_id++) {
				if (from_node_id == to_node_id) {
					matrix[from_node_id][to_node_id] = 0.;
					continue;
				}
				double tmp_weight = 0.;
				for (int index = current_index; index < end_index; index++ ) {
					tmp_weight = tmp_weight + this.all_res_cal[from_node_id][to_node_id][index];				
				}
				// make sure all eles in matrix >=0.
				if (tmp_weight > 1.) {
					tmp_weight = 1.;
				}
				matrix[from_node_id][to_node_id] = Math.log(tmp_weight)/Math.log(0.5);
			}			
		}
		
		//99, id ranges from 0 to 98
		int self_node_id = this.getHost().getAddress();
		//2. Djk Alg., calculate the shortest distance
		double[] dis = new double[this.number_nodes];
		int[] pre = new int[this.number_nodes];
		boolean[] vis = new boolean[this.number_nodes];
		for (int tmp_node_id = 0; tmp_node_id < this.number_nodes; tmp_node_id++) {
			dis[tmp_node_id] = matrix[self_node_id][tmp_node_id];
			pre[tmp_node_id] = self_node_id;
			vis[tmp_node_id] = false;
		}
		vis[self_node_id] = true;
		
		int count = 0;
		while (count != this.number_nodes) {
			// find the shortest path from the current view, dis[]
			int min_dis_id = -1;
			double min_dis = Double.MAX_VALUE;
			for (int i=0; i<this.number_nodes; i++) {
				if ((vis[i] == false) && (dis[i]<min_dis)) {
					min_dis = dis[i];
					min_dis_id = i;
				}
			}
			/**
			 * if no any way from visited node to another node 
			 */
			if (min_dis_id == -1) {
				break;
			}
			vis[min_dis_id] = true;
			count++;
			for (int i=0; i<this.number_nodes; i++) {
				if ( (vis[i]==false) && (matrix[min_dis_id][i] != Double.MAX_VALUE) 
						&& (dis[min_dis_id] + matrix[min_dis_id][i] < dis[i])){
					dis[i] = dis[min_dis_id] + matrix[min_dis_id][i];
					pre[i] = min_dis_id;
				}
			}
		}
		
		double res_tmp = dis[m.getTo().getAddress()];
		double path_value = Math.exp(res_tmp * Math.log(0.5));
		//construct the path
		List<Integer> path = new ArrayList<> ();
		int current_node = m.getTo().getAddress();
		path.add(current_node);

		while (current_node != self_node_id) {
			// can not 
			if (pre[current_node] == -1) {
				path.clear();
				break;
			}
			path.add(0, pre[current_node]);
			current_node = pre[current_node];
		} 
		return path_value;
	}
}
