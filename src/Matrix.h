class Matrix {
public:
  int rows, cols;
  std::vector<std::vector<int>> data;

  Matrix() {}
  Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<int>(c, 0)) {}
  Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}

  Matrix& operator=(const Matrix& other) {
	if (this != &other) {
	  rows = other.rows;
	  cols = other.cols;
	  data = other.data;
	}
	return *this;
  }

  bool empty(){
	return data.empty();
  }
  std::vector<int>& operator[](int i) {
	return data[i];
  }

  const std::vector<int>& operator[](int i) const {
	return data[i];
  }

  void print(string s) const {
	cout << s << endl;
	print();
  }
  void print() const {
	for (const auto& row : data) {
	  for (int val : row) {
		std::cout << std::hex << std::uppercase << std::setw(2) << val; // 16進数表示
	  }
	  std::cout << "\n";
	}
	std::cout << std::dec; // 10進数表示に戻す
  }

  Matrix transpose() const {
	Matrix result(cols, rows);
	for (int i = 0; i < rows; ++i) {
	  for (int j = 0; j < cols; ++j) {
		result[j][i] = data[i][j];
	  }
	}
	return result;
  }

  Matrix Multiply(const Matrix& other) const {
	assert(cols == other.rows);
	Matrix result(rows, other.cols);
	cout << "H20" << endl;
	for (int i = 0; i < rows; ++i) {
	  printf("%d/%d\n",i,rows);
	  for (int j = 0; j < other.cols; ++j) {
		for (int k = 0; k < cols; ++k) {
		  result[i][j] = ADDGF[result[i][j]][MULGF[data[i][k]][other[k][j]]];
		}
	  }
	}
	return result;
  }
  
  bool is_all_zero(){
	for (int i = 0; i < rows; ++i) {
	  for (int j = 0; j < cols; ++j) {
		if(data[i][j])return false;
	  }
	}
	return true;
  }

  std::vector<int> Multiply(const std::vector<int>& vec) const {
	cout << cols << " " << vec.size() << endl;
	assert(cols == vec.size());
	std::vector<int> result(rows, 0);
	for (int i = 0; i < rows; ++i) {
	  for (int j = 0; j < cols; ++j) {
		result[i] = ADDGF[result[i]][MULGF[data[i][j]][vec[j]]];
	  }
	}
	return result;
  }

  //  第 i 行を削除
  void removeRow(int i) {
	if (i < 0 || i >= rows) {
	  throw std::out_of_range("Invalid row index");
	}
	data.erase(data.begin() + i);
	--rows;
  }
  //  第 j1 列と j2 列を入れ替え
  void swapColumns(int j1, int j2) {
	if (j1 < 0 || j1 >= cols || j2 < 0 || j2 >= cols) {
	  throw std::out_of_range("Invalid column index");
	}
	for (int i = 0; i < rows; ++i) {
	  std::swap(data[i][j1], data[i][j2]);
	}
  }};
