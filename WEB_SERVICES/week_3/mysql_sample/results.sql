use test
set names utf8;

-- 1. Выбрать все товары (все поля)
select * from product

-- 2. Выбрать названия всех автоматизированных складов
select DISTINCT name from store where (is_automated = 1)

-- 3. Посчитать общую сумму в деньгах всех продаж
select sum(total) from sale

-- 4. Получить уникальные store_id всех складов, с которых была хоть одна продажа
select DISTINCT store_id from sale where (quantity > 0)


-- 5. Получить уникальные store_id всех складов, с которых не было ни одной продажи
select store.store_id from store left join sale using (store_id) where (quantity is NULL)

-- 6. Получить для каждого товара название и среднюю стоимость единицы товара avg(total/quantity), если товар не продавался, он не попадает в отчет.
select name, AVG(total / quantity) from product left join sale using (product_id) where (quantity is not NULL) group by name

-- 7. Получить названия всех продуктов, которые продавались только с единственного склада
select product.name from sale join product using(product_id) group by product_id HAVING COUNT(distinct store_id) = 1

-- 8. Получить названия всех складов, с которых продавался только один продукт
select store.name from sale join store using(store_id) group by store_id HAVING COUNT(distinct product_id) = 1

-- 9. Выберите все ряды (все поля) из продаж, в которых сумма продажи (total) максимальна (равна максимальной из всех встречающихся)
select * from sale where total in (select MAX(total) from sale)

-- 10. Выведите дату самых максимальных продаж, если таких дат несколько, то самую раннюю из них
select * from sale where total in (select MAX(total) from sale) order by date desc limit 1
