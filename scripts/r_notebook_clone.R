# r_notebook_clone.R
# Faithful recreation of the original R Titanic notebook logic

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(readr)
  library(purrr)
  library(xgboost)
})

set.seed(2025)

root_dir <- normalizePath("..", winslash = "/", mustWork = TRUE)
train_path <- file.path(root_dir, "data", "raw", "train.csv")
test_path <- file.path(root_dir, "data", "raw", "test.csv")

train <- read_csv(train_path, show_col_types = FALSE)
test <- read_csv(test_path, show_col_types = FALSE)

test$Survived <- NA_integer_
train$dataset <- "train"
test$dataset <- "test"

full <- bind_rows(train, test)

full <- full %>%
  mutate(
    Title = case_when(
      str_detect(Name, "Master") ~ "boy",
      Sex == "female" ~ "woman",
      TRUE ~ "man"
    ),
    FamilySize = coalesce(SibSp, 0L) + coalesce(Parch, 0L) + 1L,
    TicketFreq = ave(Ticket, Ticket, FUN = length),
    FareAdj = Fare / TicketFreq,
    Surname = str_extract(Name, "^[^,]+"),
    TicketMod = str_replace(Ticket, ".$", "X"),
    Embarked = replace_na(Embarked, "S")
  )

full$Age[is.na(full$Age)] <- median(full$Age, na.rm = TRUE)
full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm = TRUE)

full$GroupId <- paste(full$Surname, full$Pclass, full$TicketMod, full$Fare, full$Embarked, sep = "-")
full$GroupId[full$Title == "man"] <- "noGroup"

full <- full %>%
  group_by(GroupId) %>%
  mutate(GroupFreq = n()) %>%
  ungroup()

full$GroupId[full$GroupFreq <= 1] <- "noGroup"

train_full <- full %>% filter(dataset == "train")
test_full <- full %>% filter(dataset == "test")

wcg_rates <- train_full %>%
  filter(GroupId != "noGroup") %>%
  group_by(GroupId) %>%
  summarise(
    size = n(),
    survival_rate = mean(Survived, na.rm = TRUE),
    .groups = "drop"
  )

wcg_all_survive <- wcg_rates %>% filter(survival_rate == 1) %>% pull(GroupId)
wcg_all_die <- wcg_rates %>% filter(survival_rate == 0) %>% pull(GroupId)

initial_pred <- ifelse(test_full$Sex == "female", 1L, 0L)
initial_pred[test_full$GroupId %in% wcg_all_survive] <- 1L
initial_pred[test_full$GroupId %in% wcg_all_die] <- 0L

male_train <- train_full %>% filter(Title == "man")
female_train <- train_full %>% filter(Title == "woman", FamilySize == 1, GroupId == "noGroup")

male_features <- male_train %>%
  transmute(
    x1 = Fare / (TicketFreq * 10),
    x2 = FamilySize + Age / 70
  )

male_model <- NULL
if (nrow(male_features) > 0) {
  male_matrix <- xgb.DMatrix(as.matrix(male_features), label = male_train$Survived)
  male_params <- list(
    objective = "binary:logistic",
    eval_metric = "error",
    max_depth = 5,
    eta = 0.1,
    gamma = 0.1,
    colsample_bytree = 1.0,
    min_child_weight = 1
  )
  male_model <- xgb.train(
    params = male_params,
    data = male_matrix,
    nrounds = 500,
    verbose = 0
  )
}

female_features <- female_train %>%
  transmute(
    x1 = FareAdj / 10,
    x2 = Age / 15
  )

female_model <- NULL
if (nrow(female_features) > 0) {
  female_matrix <- xgb.DMatrix(as.matrix(female_features), label = female_train$Survived)
  female_params <- list(
    objective = "binary:logistic",
    eval_metric = "error",
    max_depth = 5,
    eta = 0.1,
    gamma = 0.1,
    colsample_bytree = 1.0,
    min_child_weight = 1
  )
  female_model <- xgb.train(
    params = female_params,
    data = female_matrix,
    nrounds = 500,
    verbose = 0
  )
}

final_pred <- initial_pred

if (!is.null(male_model)) {
  male_test <- test_full %>% filter(Title == "man")
  if (nrow(male_test) > 0) {
    male_test_mat <- male_test %>%
      transmute(
        x1 = Fare / (TicketFreq * 10),
        x2 = FamilySize + Age / 70
      ) %>%
      as.matrix()
    male_probs <- predict(male_model, male_test_mat)
    male_ids <- which(test_full$Title == "man")
    male_pred <- as.integer(male_probs >= 0.9)
    final_pred[male_ids] <- male_pred
  }
}

if (!is.null(female_model)) {
  female_test <- test_full %>% filter(Title == "woman", FamilySize == 1, GroupId == "noGroup")
  if (nrow(female_test) > 0) {
    female_test_mat <- female_test %>%
      transmute(
        x1 = FareAdj / 10,
        x2 = Age / 15
      ) %>%
      as.matrix()
    female_probs <- predict(female_model, female_test_mat)
    mask <- test_full$Title == "woman" & test_full$FamilySize == 1 & test_full$GroupId == "noGroup"
    final_pred[mask] <- as.integer(female_probs > 0.08)
  }
}

submission <- tibble(
  PassengerId = test_full$PassengerId,
  Survived = final_pred
)

output_path <- file.path(root_dir, "outputs", "submissions", "submission_r_notebook_clone.csv")
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_csv(submission, output_path)

cat("Saved submission to", output_path, "\n")
