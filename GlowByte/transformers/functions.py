def classify_weather_condition(text, weather_dict):
    if pd.notna(text):
        text = str(text)
        tokens = word_tokenize(text, language="russian")
        tokens = [
            word for word in tokens if word.lower() not in stopwords.words("russian")
        ]
        for token in tokens:
            if token in weather_dict:
                return weather_dict[token]
        return text
    return None


class PositionalEncoder(nn.Module):
    """
    Источник:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512
    ):
        """
        Args:
            dropout: регуляризация
            max_seq_len: длина последовательности
            d_model: Размерность вывода подслоев в модели
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_seq_len).unsqueeze(1)
        exp_input = torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        div_term = torch.exp(exp_input)
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Тензор, размерность [batch_size, enc_seq_len, dim_val]
        """

        add = self.pe[: x.size(1), :].squeeze(1)
        x = x + add

        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        dec_seq_len: int,
        max_seq_len: int,
        out_seq_len: int = 24,
        dim_val: int = 512,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads: int = 8,
        dropout_encoder: float = 0.2,
        dropout_decoder: float = 0.2,
        dropout_pos_enc: float = 0.2,
        dim_feedforward_encoder: int = 2048,
        dim_feedforward_decoder: int = 2048,
    ):
        """
        Args:
            input_size: int, количество входных переменных. 1, если одномерная последовательность.
             dec_seq_len: int, длина входной последовательности, подаваемой в декодер.
             max_seq_len: int, длина самой длинной последовательности, которую будет использовать модель.
             out_seq_len: int, длина выходных данных модели (т. е. целевая
                          длина последовательности)
             dim_val: int, он же d_model. Все подслои модели создают
                      выходные данные измерения dim_val
             n_encoder_layers: int, количество сложенных слоев кодера в енкодере.
             n_decoder_layers: int, количество сложенных слоев кодера в декодере.
             n_heads: int, количество attention heads (также известных как параллельные слои внимания)
             dropout_encoder: float, регуляризатор енкодера
             dropout_decoder: float, регуляризатор декодера
             dropout_pos_enc: float, регуляризатор позиционного энкодера
             dim_feedforward_encoder: int, количество нейронов в линейном слое енкодера
             dim_feedforward_decoder: int, количество нейронов в линейном слое декодера
        """

        super().__init__()

        self.dec_seq_len = dec_seq_len
        print("input_size is: {}".format(input_size))
        print("dim_val is: {}".format(dim_val))
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, out_features=dim_val
        )
        self.decoder_input_layer = nn.Linear(
            in_features=input_size, out_features=dim_val
        )
        self.linear_mapping = nn.Linear(
            in_features=out_seq_len * dim_val, out_features=out_seq_len
        )
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val, dropout=dropout_pos_enc, max_seq_len=max_seq_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None
        )

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None
    ) -> Tensor:
        """
        Args:
             src: выходная последовательность кодера. Форма: (S,E) для непакетированного ввода,
                  (S, N, E), если пакет_first=False, или (N, S, E), если пакет_first=True,
                  где S — длина исходной последовательности, N — размер пакета, а E — номер функции.
             tgt: последовательность для декодера. Форма: (T,E) для непакетированного ввода, (T, N, E)(T,N,E),
             если пакет_first=False или (N, T, E), если пакет_first=True, где T — длина целевой последовательности,
             N — размер партии, E — номер функции.
             src_mask: маска для последовательности src, чтобы модель не могла использовать точки данных
             из целевой последовательности.
             tgt_mask: маска для последовательности tgt, предотвращающая использование моделью точек данных
             из целевой последовательности.
        """
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)
        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(
            tgt=decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask
        )
        decoder_output = self.linear_mapping(decoder_output.flatten(start_dim=1))
        return decoder_output


class TransformerDataset(Dataset):
    """
    Класс набора данных, используемый для трансформеров.

    """

    def __init__(
        self,
        data: torch.tensor,
        indices: list,
        enc_seq_len: int,
        dec_seq_len: int,
        target_seq_len: int,
    ) -> None:
        """
        Args:
            data: тензор, весь train, последовательность проверочных или тестовых данных. Если одномерный,
                data.size() будет [количество образцов, количество переменных], где количество переменных
                будет равно 1 + количество экзогенные переменные. Количество экзогенных переменных будет 0
                если последоватеьность одномерная.
             indices: список кортежей. Каждый кортеж состоит из двух элементов:
                      1) начальный индекс подпоследовательности
                      2) конечный индекс подпоследовательности.
                      Подпоследовательность позже разбивается на src, trg и trg_y.
             enc_seq_len: int, желаемая длина входной последовательности.
             target_seq_len: int, желаемая длина целевой последовательности (выходные данные модели)
             target_idx: позиция индекса целевой переменной в данных.
        """

        super().__init__()
        self.indices = indices
        self.data = data
        print("From get_src_trg: data size = {}".format(data.size()))
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        Возвращает кортеж из трех элементов:
         1) src (вход энкодера)
         2) trg (вход декодера)
         3) trg_y (цель)
        """
        start_idx = self.indices[index][0]
        end_idx = self.indices[index][1]
        sequence = self.data[start_idx:end_idx]
        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len,
        )
        return src, trg, trg_y

    def get_src_trg(
        self,
        sequence: torch.Tensor,
        enc_seq_len: int,
        dec_seq_len: int,
        target_seq_len: int,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Ненерирует последовательности src (вход кодера), trg (вход декодера) и trg_y (цель).
        Args:
            sequence: тензор, одномерный тензор длины n, где n = входная длина кодера + длина целевой последовательности
            enc_seq_len: int, желаемая длина входа в энкодер трансформера
            target_seq_len: int, желаемая длина целевой последовательности (той, с которой сравниваются выходные данные модели)
        Return:
            src: tensor, 1D
            trg: tensor, 1D
            trg_y: tensor, 1D

        """
        assert (
            len(sequence) == enc_seq_len + target_seq_len
        ), "Sequence length does not equal (input length + target length)"
        src = sequence[:enc_seq_len]
        trg = sequence[enc_seq_len - 1 : len(sequence) - 1]
        assert (
            len(trg) == target_seq_len
        ), "Length of trg does not match target sequence length"
        trg_y = sequence[-target_seq_len:]
        assert (
            len(trg_y) == target_seq_len
        ), "Length of trg_y does not match target sequence length"

        return (
            src,
            trg,
            trg_y.squeeze(-1),
        )  # reshape from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]


def generate_square_subsequent_mask(dim1: int, dim2: int, dim3: int) -> Tensor:
    """
    Генерирует верхнетреугольную матрицу -inf заполненную нулями.
    Источник:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, batch_size * n_heads
        dim2: инт. Для маскировки src и trg это должна быть длина целевой последовательности.
        dim3: инт. Для маскировки src это должна быть длина последовательности кодера.
               Для маскировки trg это должна быть длина целевой последовательности.
    Return:
        Tensor of shape [dim1, dim2, dim3]
    """
    return torch.triu(torch.ones(dim1, dim2, dim3) * float("-inf"), diagonal=1)


def get_indices_input_target(
    num_obs, input_len, step_size, forecast_horizon, target_len
):
    """
    Создаёт все начальные и конечные индексные позиции всех подпоследовательностей.
     Индексы будут использоваться для разделения данных на подпоследовательности, на которых
     модели будут обучены.
     Возвращает кортеж из четырех элементов:
     1) Индексная позиция первого элемента, который будет включен во входную последовательность.
     2) Индексная позиция последнего элемента, который будет включен во входную последовательность.
     3) Индексная позиция первого элемента, который будет включен в целевую последовательность.
     4) Индексная позиция последнего элемента, который будет включен в целевую последовательность.

    Args:
        num_obs (int): количество наблюдений во всем наборе данных, для которых необходимо создать индексы.
         input_len (int): длина входной последовательности (подпоследовательность всей последовательности данных)
         Step_size (int): Размер каждого шага при прохождении последовательности данных.
                         Если 1, первая подпоследовательность будет иметь индекс 0-input_len,
                         а следующая будет 1-input_len.
         forecast_horizon (int): На сколько позиций индекса цель находится дальше от последней позиции
                                 индекса входной последовательности? Если forecast_horizon=1 и входная
                                 последовательность — data[0:10], целью будет data[11:taget_len].
         target_len (int): длина целевой/выходной последовательности.
    """

    input_len = round(input_len)  # just a precaution
    start_position = 0
    stop_position = num_obs - 1  # because of 0 indexing
    subseq_first_idx = start_position
    subseq_last_idx = start_position + input_len
    target_first_idx = subseq_last_idx + forecast_horizon
    target_last_idx = target_first_idx + target_len
    print("target_last_idx is {}".format(target_last_idx))
    print("stop_position is {}".format(stop_position))
    indices = []
    while target_last_idx <= stop_position:
        indices.append(
            (subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx)
        )
        subseq_first_idx += step_size
        subseq_last_idx += step_size
        target_first_idx = subseq_last_idx + forecast_horizon
        target_last_idx = target_first_idx + target_len

    return indices


def get_indices_entire_sequence(
    data: pd.DataFrame, window_size: int, step_size: int
) -> list:
    """
    Создаёт все начальные и конечные позиции индексов, необходимые для создания подпоследовательностей.
     Возвращает список кортежей. Каждый кортеж представляет собой (start_idx, end_idx) подпоследовательности.
     Эти кортежи следует использовать для разделения набора данных на подпоследовательности.
     Эти подпоследовательности затем следует передать в функцию, которая делит их на входную и целевую последовательности

    Args:
        num_obs (int): Количество наблюдений (time steps)
        window_size (int): (input_sequence_length + target_sequence_length)
        step_size (int): Размер каждого шага при прохождении последовательности данных window_size.
                        Если 1, первая подпоследовательность будет [0:window_size], а следующая — [1:window_size].
    Return:
        indices: список кортежей
    """
    stop_position = len(data) - 1  # 1- because of 0 indexing
    subseq_first_idx = 0
    subseq_last_idx = window_size
    indices = []
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_last_idx += step_size

    return indices


def read_data(
    data_dir: Union[str, Path] = "/kaggle/input/glowbyte/",
    timestamp_col_name: str = "date",
    norm: bool = True,
) -> pd.DataFrame:
    """
    Читает данные из csv и возвращает pd.Dataframe
    Args:
        data_dir: путь к директории, где находится файл
        timestamp_col_name: str, столбец временной шкалы
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    if len(csv_files) > 1:
        raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")

    data_path = csv_files[0]
    print("Reading file in {}".format(data_path))
    df = pd.read_csv(
        data_path,
        parse_dates=[timestamp_col_name],
        index_col=[timestamp_col_name],
        infer_datetime_format=True,
        low_memory=False,
    )
    if is_ne_in_df(df):
        raise ValueError("DataFrame contains 'n/e' values. These must be handled")
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = pd.to_timedelta(df["time"], unit="h")
    df["ds"] = df["date"] + df["time"]
    df = df.drop(["date", "time"], axis=1)
    df.rename(columns={"target": "y"}, inplace=True)
    data = to_numeric_and_downcast_data(df)
    if norm == True:
        initialize_norm = standardize_data()
        data = initialize_norm.to_normalize(data)
    data.sort_values(by=["ds"], inplace=True)

    return initialize_norm, data


class standardize_data:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def to_normalize(self, data):
        data[["y"]] = self.scaler.fit_transform(data[["y"]].values)
        return data

    def to_reverse_normalization(self, data):
        data[["y"]] = self.scaler.inverse_transform(data[["y"]].values)
        return data


def is_ne_in_df(df: pd.DataFrame):
    for col in df.columns:
        true_bool = df[col] == "n/e"
        if any(true_bool):
            return True
    return False


def to_numeric_and_downcast_data(df: pd.DataFrame):
    fcols = df.select_dtypes("float").columns
    icols = df.select_dtypes("integer").columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
    df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

    return df


def prepare_data(data1, batch_size):
    data_indices = get_indices_entire_sequence(
        data=data1, window_size=window_size, step_size=1
    )
    data1 = TransformerDataset(
        data=torch.tensor(data1[input_variables].values).float(),
        indices=data_indices,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        target_seq_len=output_sequence_length,
    )

    return DataLoader(data1, batch_size)


def input_data(seq, ws):
    out = []
    L = len(seq)
    for i in range(L - ws):
        window = seq[i : i + ws]
        label = seq[i + ws : i + ws + 1]
        out.append((window, label))
    return out


class LSTMnetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = (
            torch.zeros(1, 1, self.hidden_size),
            torch.zeros(1, 1, self.hidden_size),
        )

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f"{item:>6}")
    print(f"______\n{sum(params):>6}")
