library(jsonlite)

save_parameters <- function(parameters, file){
    jsonlite::write_json(parameters, file, dataframe="columns", pretty = TRUE)
}

load_parameters <- function(file){
    return(jsonlite::fromJSON(file))
}

# Example usage
# df = read.csv("simulated-.csv") # path from TSbench root : data/test_csv
# write.csv(df, "test.csv", row.names=FALSE) # Optional : keep the data
# df = df[6] # fit on last observations with no Na
# params = train_ruGARCH(df, 1, 1)
# df <- as.data.frame(t(params))
# save_parameters(df, "test.json")
# params = load_parameters("test.json")
# reloaded_params <- as.data.frame(params) # for the example
