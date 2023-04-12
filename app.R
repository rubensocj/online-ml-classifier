# Online ML Classifier
# Rubens O. da Cunha Júnior

# Load packages ----
library(shiny)
library(shinyjs)
library(shinythemes)
library(shinyBS)
library(shinydashboard)
library(DT)
library(e1071)
library(rpart)
library(randomForest)

# Prepare environment ----
set.seed(123)

n_models <- c(
  "Support Vector Machine",
  "Naive Bayes",
  "k-Nearest Neighbors",
  "Decision Trees",
  "Random Forest")

# Functions ----
evaluation <- function(target, predicted) {
  cm <- table(target, predicted)

  accuracy <- sum(diag(cm)) / sum(cm)
  precision <- diag(cm) / apply(cm, 2, sum)
  recall <- diag(cm) / apply(cm, 1, sum)
  f1 <- 2*precision*recall / (precision+recall)

  p <- apply(cm, 1, sum) / sum(cm)
  q <- apply(cm, 2, sum) / sum(cm)
  expected_accuracy <- sum(p*q)
  kappa <- (accuracy - expected_accuracy) / (1 - expected_accuracy)

  return(list("Accuracy" = accuracy, "Precision" = precision,
              "Recall" = recall, "F-1" = f1, "Kappa Statistic" = kappa,
              "Confusion Matrix" = cm))
}

disable_input <- function() {
  shinyjs::disable("y")
  shinyjs::disable("type_x")
  shinyjs::disable("x")
  shinyjs::disable("run_modelling")
}
enable_input <- function() {
  shinyjs::enable("y")
  shinyjs::enable("type_x")
  shinyjs::enable("run_modelling")
}

# UI ----
ui <- dashboardPage(
  skin = "black",
  title = "Online ML Classifier",

  header = dashboardHeader(title = "Online ML Classifier"),
  sidebar = dashboardSidebar(
    sidebarMenu(
      menuItem(
        text = "Home",
        tabName = "home",
        icon = icon("home")
      ),
      menuItem(
        text = "Input/Feature Selection",
        tabName = "input",
        icon = icon("upload")
      ),
      menuItem(
        text = "Model",
        tabName = "model",
        icon = icon("circle-nodes")
      ),
      menuItem(
        text = "Predict",
        tabName = "predict",
        icon = icon("chart-simple")
      ),
      menuItem(
        text = "Help",
        tabName = "help",
        icon = icon("question")
      )
    )
  ), # end dashboardSideBar

  body = dashboardBody(

    shinyjs::useShinyjs(),

    tabItems(
      tabItem(
        tabName = "home",
        includeHTML("www/home.html")
      ), # end tabItem

      tabItem(
        tabName = "input",

        fluidRow(
          box(
            title = "Input selection",
            width = "12",
            fileInput(
              inputId = "file_train",
              label = "Load training data",
              accept = c(".csv",".txt",".xlsx",".xls"),
              multiple = FALSE
            ),
            actionButton(
              inputId = "show_training",
              label = "Show Table",
              icon = icon("table")
            ),
            bsModal(
              id = "modal_training",
              title = "Training data",
              trigger = "show_training",
              size = "large",
              DT::dataTableOutput(outputId = "table")
            )
          ) # end box
        ), # end fluidRow

        fluidRow(
          box(
            title = "Feature selection",
            width = "12",
            selectInput(
              inputId = "y",
              label = "Dependent variable",
              choices = NULL,
              selected = NULL,
              multiple = FALSE
            ),
            radioButtons(
              inputId = "type_x",
              label = "Independent variable",
              choices = list("Use all" = 1, "Custom selection" = 2),
              selected = 1,
              inline = TRUE
            ),
            selectInput(
              inputId = "x",
              label = NULL,
              choices = NULL,
              selected = NULL,
              multiple = TRUE
            )
          ) # end box
        ), # end fluidRow

        includeHTML("www/contato.html")
      ), # end tabItem

      tabItem(
        tabName = "model",

        fluidRow(
          box(
            title = "Modelling",
            width = "12",
            actionButton(
              inputId = "run_modelling",
              label = "Run"
            )
          )
        ), # end fluidRow

        fluidRow(
          tabBox(
            title = "Results",
            side = "left",
            id = "tabset1",
            width = "12",
            selected = "Description",
            tabPanel(
              "Description",
              htmlOutput(outputId = "lbl_formula"),
              verbatimTextOutput(outputId = "formula")
            ), # end tabPanel

            tabPanel(
              "Model Summary",
              htmlOutput(outputId = "lbl_summary_svm"),
              verbatimTextOutput(outputId = "summary_svm"),

              htmlOutput(outputId = "lbl_summary_nb"),
              verbatimTextOutput(outputId = "summary_nb"),

              htmlOutput(outputId = "lbl_summary_knn"),
              verbatimTextOutput(outputId = "summary_knn"),

              htmlOutput(outputId = "lbl_summary_tr"),
              verbatimTextOutput(outputId = "summary_tr"),

              htmlOutput(outputId = "lbl_summary_rf"),
              verbatimTextOutput(outputId = "summary_rf")
            ), # end tabPanel

            tabPanel(
              "Statistics",
              htmlOutput(outputId = "lbl_accuracy"),
              tableOutput(outputId = "accuracy"),

              htmlOutput(outputId = "lbl_kappa"),
              tableOutput(outputId = "kappa"),

              htmlOutput(outputId = "lbl_precision"),
              tableOutput(outputId = "precision"),

              htmlOutput(outputId = "lbl_recall"),
              tableOutput(outputId = "recall"),

              htmlOutput(outputId = "lbl_f1"),
              tableOutput(outputId = "f1")
            ) # end tabPanel
          ) # end tabBox
        ), # end fluidRow

        includeHTML("www/contato.html")
      ), # end tabItem

      tabItem(
        tabName = "predict",

        fluidRow(
          box(
            title = "Input selection",
            width = "12",
            fileInput(
              inputId = "file_test",
              label = "Load test data",
              accept = c(".csv",".txt",".xlsx",".xls"),
              multiple = FALSE
            ),
            actionButton(
              inputId = "show_test",
              label = "Show Table",
              icon = icon("table")
            ),
            bsModal(
              id = "modal_test",
              title = "Test data",
              trigger = "show_test",
              size = "large",
              DT::dataTableOutput(outputId = "table_test")
            )
          ) # end box
        ), # end fluidRow

        fluidRow(
          box(
            title = "Predict",
            width = "12",
            actionButton(
              inputId = "run_predicting",
              label = "Run"
            )
          ) # end box
        ), # end fluidRow

        fluidRow(
          box(
            title = "Predictions",
            width = "12",
            collapsible = TRUE,
            htmlOutput(outputId = "lbl_predictions"),
            DT::DTOutput(outputId = "table_predictions")
          ) # end box
        ), # end fluidRow

        includeHTML("www/contato.html")
      ), # end tabItem

      tabItem(
        tabName = "help",
        includeHTML("www/help.html"),
        includeHTML("www/contato.html")
      ) # end tabItem
    ) # end tabItems
  ) # end dashboardBody
) # end dashboardPage

# Server ----
server <- function(input, output, session) {
  values <- reactiveValues(
    train_data = NULL,
    train_cols = NULL,
    test_data = NULL,
    predictions = NULL
  )

  observeEvent(input$file_train, {
    if(is.null(input$file_train)) {
      disable_input()
    } else {
      # Read file and update reactive values
      values$train_data <- read.csv(file = input$file_train$datapath,
                                    stringsAsFactors = TRUE)
      values$train_cols <- colnames(values$train_data)

      # Update widget
      updateSelectInput(session, "y", selected = values$train_cols[1], choices = values$train_cols)
      updateSelectInput(session, "x", selected = "", choices = values$train_cols[-1])

      enable_input()
    }
  }, ignoreNULL = F)

  # Input selection: select y variable
  observeEvent(input$y, {
    train_cols_upd <- values$train_cols
    train_cols_upd <- train_cols_upd[!train_cols_upd %in% input$y]
    updateSelectInput(session, "x", selected = "",
                      choices = train_cols_upd)
  }, ignoreNULL = T)

  # Radio button: type of x variable
  observeEvent(input$type_x, {
    if(input$type_x == 1) {
      train_cols_upd <- values$train_cols
      train_cols_upd <- train_cols_upd[!train_cols_upd %in% input$y]
      updateSelectInput(session, "x", selected = "",
                        choices = train_cols_upd)
      shinyjs::disable("x")
    } else if(input$type_x == 2) {
      shinyjs::enable("x")
    }
  }, ignoreNULL = T)

  # Action button: run modelling
  models <- reactive({
    req(c(input$y, input$x))

    # Target variable
    y <- input$y

    # Explanatory variables
    if(is.null(input$x)) {
      x <- "."
    } else {
      x <- paste(input$x, collapse = "+")
    }

    # Formula
    f <- formula(paste0(y, "~", x))

    # Modelling
    mod_svm <- e1071::svm(formula = f, data = values$train_data, type = 'C-classification')
    print("svm ok")
    mod_nb <- e1071::naiveBayes(formula = f, data = values$train_data)
    print("nb ok")
    mod_knn <- e1071::gknn(formula = f, data = values$train_data)
    print("knn ok")
    mod_tr <- rpart::rpart(formula = f, data = values$train_data)
    print("tr ok")
    mod_rf <- randomForest::randomForest(formula = f, data = values$train_data)
    print("rf ok")
    mod <- list("svm" = mod_svm, "nb" = mod_nb, "knn" = mod_knn,
                "tr" = mod_tr, "rf" = mod_rf)

    # Fitted values
    fit_svm <- predict(mod_svm, values$train_data[, !colnames(values$train_data) %in% y], type = 'class')
    fit_nb <- predict(mod_nb, values$train_data[, !colnames(values$train_data) %in% y], type = 'class')
    fit_knn <- predict(mod_knn, values$train_data[, !colnames(values$train_data) %in% y], type = 'class')
    fit_tr <- predict(mod_tr, values$train_data[, !colnames(values$train_data) %in% y], type = 'class')
    fit_rf <- predict(mod_rf, values$train_data[, !colnames(values$train_data) %in% y], type = 'class')
    fit <- list("svm" = fit_svm, "nb" = fit_nb, "knn" = fit_knn,
                 "tr" = fit_tr, "rf" = fit_rf)

    # Performance evaluation
    eval_svm <- evaluation(values$train_data[, y], fit_svm)
    eval_nb <- evaluation(values$train_data[, y], fit_nb)
    eval_knn <- evaluation(values$train_data[, y], fit_knn)
    eval_tr <- evaluation(values$train_data[, y], fit_tr)
    eval_rf <- evaluation(values$train_data[, y], fit_rf)
    eval <- list("svm" = eval_svm, "nb" = eval_nb, "knn" = eval_knn,
                "tr" = eval_tr, "rf" = eval_rf)

    return(list("models" = mod, "fitted" = fit, "eval" = eval, "formula" = f))
  })

  observeEvent(input$run_modelling, {
    req(models())
    m <- models()

    # Results
    df_accuracy <- cbind(
      "Model" = n_models,
      "Value" = format(lapply(m$eval, "[[", "Accuracy"), digits = 4))

    df_kappa <- cbind(
      "Model" = n_models,
      "Value" = format(lapply(m$eval, "[[", "Kappa Statistic"), digits = 4))

    df_precision <- cbind(
      "Model" = n_models,
      format(do.call("rbind", lapply(m$eval, "[[", "Precision")), digits = 4))

    df_recall <- cbind(
      "Model" = n_models,
      format(do.call("rbind", lapply(m$eval, "[[", "Recall")), digits = 4))

    df_f1 <- cbind(
      "Model" = n_models,
      format(do.call("rbind", lapply(m$eval, "[[", "F-1")), digits = 4))

    # Render Outputs
    # Description
    output$lbl_formula <- renderUI(h4("Formula"))
    output$formula <- renderPrint({
      deparse(m$formula)
    })

    # Summary
    output$lbl_summary_svm <- renderUI(h4("Support Vector Machine"))
    output$summary_svm <- renderPrint({
      print(summary(m$models$svm))
      cat("Confusion Matrix:\n")
      print(m$eval$svm$`Confusion Matrix`)
    })

    output$lbl_summary_nb <- renderUI(h4("Naive Bayes"))
    output$summary_nb <- renderPrint({
      print(m$models$nb)
      cat("Confusion Matrix:\n")
      print(m$eval$nb$`Confusion Matrix`)
    })

    output$lbl_summary_knn <- renderUI(h4("k-Nearest Neighbors"))
    output$summary_knn <- renderPrint({
      cat("Call:\n")
      print(m$models$knn$call)
      cat(paste("k =", m$models$knn$k, "\n\n"))
      cat("Confusion Matrix:\n")
      print(m$eval$knn$`Confusion Matrix`)
    })

    output$lbl_summary_tr <- renderUI(h4("Decision Trees"))
    output$summary_tr <- renderPrint({
      print(summary(m$models$tr))
      cat("\n")
      cat("Confusion Matrix:\n")
      print(m$eval$tr$`Confusion Matrix`)
    })

    output$lbl_summary_rf <- renderUI(h4("Random Forest"))
    output$summary_rf <- renderPrint({
      print(m$models$rf)
      cat("\n")
      cat("Confusion Matrix:\n")
      print(m$eval$rf$`Confusion Matrix`)
    })

    # Statistics
    output$lbl_accuracy <- renderUI(h4("Accuracy"))
    output$accuracy <- renderTable(df_accuracy, striped = TRUE)

    output$lbl_kappa <- renderUI(h4("Kappa"))
    output$kappa <- renderTable(df_kappa, striped = TRUE)

    output$lbl_precision <- renderUI(h4("Precision"))
    output$precision <- renderTable(df_precision, striped = TRUE)

    output$lbl_recall <- renderUI(h4("Recall"))
    output$recall <- renderTable(df_recall, striped = TRUE)

    output$lbl_f1 <- renderUI(h4("F-1"))
    output$f1 <- renderTable(df_f1, striped = TRUE)

    # Set focus to evaluation tab
    updateNavbarPage(session, inputId = "tabset_train", selected = "Evaluation")

    # Update label on evaluation tab
    output$lbl_evaluation <- renderUI(br())
  })

  # Data tab
  output$table <- DT::renderDataTable(values$train_data)

  # Evaluation tab
  output$lbl_evaluation <- renderUI({
    HTML("<br>The model evaluation will be shown here.<br>")
  })

  # Predicting Tab Panel ----
  observeEvent(input$file_test, {
    if(is.null(input$file_test)) {
      shinyjs::disable("run_predicting")
    } else {
      # Read file
      values$test_data <- read.csv(file = input$file_test$datapath,
                                   stringsAsFactors = TRUE)

      shinyjs::enable("run_predicting")
    }
  }, ignoreNULL = F)

  # Action button: run modelling
  observeEvent(input$run_predicting, {
    req(models())
    m <- models()

    # Predict
    pred_svm <- predict(m$models$svm, values$test_data, type = 'class')
    pred_nb <- predict(m$models$nb, values$test_data, type = 'class')
    pred_knn <- predict(m$models$knn, values$test_data, type = 'class')
    pred_tr <- predict(m$models$tr, values$test_data, type = 'class')
    pred_rf <- predict(m$models$rf, values$test_data, type = 'class')
    values$predictions <- data.frame(pred_svm, pred_nb, pred_knn, pred_tr, pred_rf)
    colnames(values$predictions) <- n_models

    # Update label on predictions tab
    output$lbl_predictions <- renderUI(br())
  })

  # Data tab
  output$table_test <- DT::renderDataTable(values$test_data)

  # Predictions tab
  output$lbl_predictions <- renderUI({
    HTML("<br>The predictions will be shown here.")
  })

  output$table_predictions <- DT::renderDT(
    DT::datatable(
      data = values$predictions,
      rownames = FALSE,
      extensions = 'Buttons',
      selection = "none",
      options = list(
        paging = TRUE,
        searching = TRUE,
        fixedColumns = TRUE,
        autoWidth = TRUE,
        ordering = TRUE,
        scrollY = '400px',
        dom = 'lfrtipB',
        buttons = c('csv', 'excel')
      )
    )
  )
}

# Shiny app ----
shinyApp(ui = ui, server = server)
