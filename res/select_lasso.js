function click_on_lasso_select() {
console.log("CLICKING LASSO BUTTON START - func called")

    const collection = document.getElementsByClassName("modebar-btn");
    for (var j = 0; j < collection.length; j++){
        if (collection[j].dataset.title.includes("Lasso Select")){
            ind = j
            console.log("CLICKING LASSO BUTTON START")
        }
    }
    collection[ind].click()
}
console.log("CLICKING LASSO BUTTON START - timeout start")

setTimeout(click_on_lasso_select, 2000)